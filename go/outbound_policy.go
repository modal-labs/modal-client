package modal

import (
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strings"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
)

var (
	outboundPolicyDomainRe = regexp.MustCompile(
		`^(\*\.)?([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$`,
	)
	outboundPolicyHeaderNameRe = regexp.MustCompile(`^[A-Za-z0-9!#$%&'*+.^_` + "`" + `|~-]+$`)
)

func hasInvalidHeaderValueChar(value string) bool {
	return strings.ContainsFunc(value, func(r rune) bool {
		return (r < 0x20 && r != '\t') || r == 0x7F
	})
}

// templateReferencesKey reports whether a header value template references a `$KEY`.
func templateReferencesKey(template string) bool {
	isKeyStart := func(c byte) bool {
		return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
	}
	for i := 0; i+1 < len(template); i++ {
		if template[i] != '$' {
			continue
		}
		c := template[i+1]
		if c == '$' {
			i++ // skip the escaped pair
			continue
		}
		if isKeyStart(c) {
			return true
		}
	}
	return false
}

// OutboundPolicy is a Sandbox's route-based outbound network policy.
//
// Routes are evaluated in order, and the first route that matches decides
// what happens to the traffic: it goes to its original destination. The last
// route must be the policy's only OutboundRouteAny, which matches all traffic
// no other route matched and either lets it through or blocks it.
//
//	policy := modal.OutboundPolicy{
//		AllowTLSDecryption: true,
//		Routes: []modal.OutboundRoute{
//			modal.OutboundRouteHTTPS{
//				Domains: []string{"api.example.com"},
//				HeaderReplacements: []modal.OutboundHeaderReplacement{
//					{Secret: secret, Headers: map[string]string{"Authorization": "Bearer $API_TOKEN"}},
//				},
//			},
//			modal.OutboundRouteTLS{Domains: []string{"github.com", "*.github.com"}},
//			modal.OutboundRouteIP{CIDRs: []string{"203.0.113.0/24"}},
//			modal.OutboundRouteAny{Redirect: modal.OutboundBlock{}},
//		},
//	}
type OutboundPolicy struct {
	// AllowTLSDecryption allows Modal to decrypt the Sandbox's TLS traffic on
	// routes that need it, by generating a certificate authority for the
	// Sandbox and attempting to make the Sandbox trust it. It decrypts nothing
	// on its own. OutboundRouteHTTPS requires it. It is fixed when the Sandbox
	// is created. Sandboxes that allow TLS decryption don't support memory
	// snapshots.
	AllowTLSDecryption bool
	Routes             []OutboundRoute
}

// OutboundRoute is a route in an OutboundPolicy: an OutboundRouteTLS,
// OutboundRouteHTTPS, OutboundRouteIP or OutboundRouteAny.
type OutboundRoute interface{ isOutboundRoute() }

// OutboundRouteTLS matches connections that carry TLS from the first byte,
// by the domain the client requests (SNI), on tcp/443. The connection is
// never decrypted. Traffic to these domains on other ports or transports
// isn't matched by this route.
type OutboundRouteTLS struct {
	// Domains supports "*." prefixes, which match subdomains at any depth
	// but not the domain itself: "*.example.com" matches "a.example.com" and
	// "a.b.example.com", not "example.com". A bare "*" matches all domains.
	Domains []string
}

// OutboundRouteHTTPS matches HTTPS connections by the domain the client
// requests, on tcp/443. The connection is always decrypted, which requires
// OutboundPolicy.AllowTLSDecryption, and requests whose Host doesn't match
// the requested domain are refused. Traffic to these domains on other ports
// or transports isn't matched by this route.
type OutboundRouteHTTPS struct {
	// Domains supports "*." prefixes, which match subdomains at any depth
	// but not the domain itself: "*.example.com" matches "a.example.com" and
	// "a.b.example.com", not "example.com". A bare "*" matches all domains.
	Domains []string
	// HeaderReplacements are in public beta.
	HeaderReplacements []OutboundHeaderReplacement
}

// OutboundRouteIP matches all traffic to a set of destination CIDRs, on any
// transport and port.
type OutboundRouteIP struct {
	CIDRs []string
}

// OutboundRouteAny matches all traffic that earlier routes didn't. It must
// be the last route of an OutboundPolicy.
type OutboundRouteAny struct {
	// Redirect is nil to let the traffic through to its original
	// destination; OutboundBlock blocks it.
	Redirect OutboundRouteAnyRedirect
}

func (OutboundRouteTLS) isOutboundRoute()   {}
func (OutboundRouteHTTPS) isOutboundRoute() {}
func (OutboundRouteIP) isOutboundRoute()    {}
func (OutboundRouteAny) isOutboundRoute()   {}

// OutboundHeaderReplacement replaces headers in HTTP requests on an
// OutboundRouteHTTPS.
//
// Each header is set on every matching request, so the request has exactly
// one header with that name and the given value. Copies the request already
// has are removed, whatever their case, and the header is added if the
// request has none. A route can set each header name only once, ignoring
// case.
//
// Header values support templating with keys in Secret: a "$"-prefixed key
// name is replaced with the secret value. Literal "$" characters are written
// "$$". Secret values never enter the Sandbox.
type OutboundHeaderReplacement struct {
	// Secret is a named Secret (e.g. from Client.Secrets.FromName), or nil
	// for static values.
	Secret  *Secret
	Headers map[string]string
}

// OutboundRouteAnyRedirect is a redirect an OutboundRouteAny accepts.
type OutboundRouteAnyRedirect interface{ isOutboundRouteAnyRedirect() }

// OutboundBlock blocks the traffic.
type OutboundBlock struct{}

func (OutboundBlock) isOutboundRouteAnyRedirect() {}

// derefRoute returns route with any pointer removed, or nil for a nil route.
func derefRoute(route OutboundRoute) OutboundRoute {
	switch r := route.(type) {
	case *OutboundRouteTLS:
		if r != nil {
			return *r
		}
	case *OutboundRouteHTTPS:
		if r != nil {
			return *r
		}
	case *OutboundRouteIP:
		if r != nil {
			return *r
		}
	case *OutboundRouteAny:
		if r != nil {
			return *r
		}
	default:
		return route
	}
	return nil
}

// blocks reports whether redirect, which may be a pointer, is OutboundBlock.
func blocks(redirect OutboundRouteAnyRedirect) bool {
	switch r := redirect.(type) {
	case OutboundBlock:
		return true
	case *OutboundBlock:
		return r != nil
	}
	return false
}

// validate checks the policy, returning the first violation found.
func (p *OutboundPolicy) validate() error {
	if p == nil {
		return nil
	}
	if len(p.Routes) == 0 {
		return InvalidError{Exception: "OutboundPolicy.Routes must end with an OutboundRouteAny"}
	}
	for i, route := range p.Routes {
		invalid := func(format string, args ...any) error {
			return InvalidError{Exception: fmt.Sprintf("OutboundPolicy.Routes[%d]: ", i) + fmt.Sprintf(format, args...)}
		}
		last := i == len(p.Routes)-1
		switch r := derefRoute(route).(type) {
		case OutboundRouteTLS:
			if err := validateOutboundDomains(r.Domains); err != nil {
				return invalid("%s", err)
			}
		case OutboundRouteHTTPS:
			if err := validateOutboundDomains(r.Domains); err != nil {
				return invalid("%s", err)
			}
			if !p.AllowTLSDecryption {
				return invalid("HTTPS routes require OutboundPolicy.AllowTLSDecryption")
			}
			names := map[string]bool{}
			for _, h := range r.HeaderReplacements {
				if err := validateHeaderReplacement(h.Headers, h.Secret); err != nil {
					return invalid("%s", err.(InvalidError).Exception)
				}
				for _, name := range slices.Sorted(maps.Keys(h.Headers)) {
					if names[strings.ToLower(name)] {
						return invalid("header %q is set more than once in the route", name)
					}
					names[strings.ToLower(name)] = true
				}
			}
		case OutboundRouteIP:
			if len(r.CIDRs) == 0 {
				return invalid("an IP route needs at least one CIDR")
			}
		case OutboundRouteAny:
			if !last {
				return invalid("an OutboundRouteAny must be the last route")
			}
			if r.Redirect != nil && !blocks(r.Redirect) {
				return invalid("unsupported redirect %T", r.Redirect)
			}
		case nil:
			return invalid("route is nil")
		}
		if last {
			if _, ok := derefRoute(route).(OutboundRouteAny); !ok {
				return InvalidError{Exception: "OutboundPolicy.Routes must end with an OutboundRouteAny"}
			}
		}
	}
	return nil
}

func validateOutboundDomains(domains []string) error {
	if len(domains) == 0 {
		return fmt.Errorf("a route needs at least one domain")
	}
	for _, domain := range domains {
		if domain != "*" && !outboundPolicyDomainRe.MatchString(domain) {
			return fmt.Errorf("invalid domain: %q", domain)
		}
	}
	return nil
}

// validateHeaderReplacement checks one header replacement's headers and
// Secret, returning the first violation found.
func validateHeaderReplacement(headers map[string]string, secret *Secret) error {
	if len(headers) == 0 {
		return InvalidError{Exception: "Headers must contain at least one header"}
	}
	// Sort the names for deterministic error messages.
	for _, name := range slices.Sorted(maps.Keys(headers)) {
		value := headers[name]
		if !outboundPolicyHeaderNameRe.MatchString(name) {
			return InvalidError{Exception: fmt.Sprintf("Invalid header name: %q", name)}
		}
		if hasInvalidHeaderValueChar(value) {
			return InvalidError{Exception: fmt.Sprintf("Header value for %q must not contain control characters (except tab)", name)}
		}
		if secret == nil && templateReferencesKey(value) {
			return InvalidError{Exception: fmt.Sprintf(
				"Header value for %q references a secret key, but no Secret was passed", name)}
		}
	}
	if secret != nil {
		if _, isEnvDict := secretEnvDictHydrator(secret); isEnvDict {
			return InvalidError{Exception: "Outbound policies only support named secrets" +
				" (e.g. Client.Secrets.FromName); ephemeral secrets (e.g. Client.Secrets.FromMap)" +
				" are not yet supported"}
		}
	}
	return nil
}

// secrets returns the deduplicated Secrets the policy's header replacements
// reference.
func (p *OutboundPolicy) secrets() []*Secret {
	if p == nil {
		return nil
	}
	var out []*Secret
	for _, route := range p.Routes {
		r, ok := derefRoute(route).(OutboundRouteHTTPS)
		if !ok {
			continue
		}
		for _, h := range r.HeaderReplacements {
			if h.Secret != nil && !slices.Contains(out, h.Secret) {
				out = append(out, h.Secret)
			}
		}
	}
	return out
}

// toProto converts a validated policy to the wire format, or returns nil for a
// nil policy. Referenced Secrets must be hydrated first.
func (p *OutboundPolicy) toProto() *pb.OutboundPolicy {
	if p == nil {
		return nil
	}
	routes := make([]*pb.OutboundPolicy_Route, 0, len(p.Routes))
	for _, route := range p.Routes {
		var b pb.OutboundPolicy_Route_builder
		switch r := derefRoute(route).(type) {
		case OutboundRouteTLS:
			b.Tls = pb.OutboundPolicy_TLSRoute_builder{Domains: r.Domains}.Build()
		case OutboundRouteHTTPS:
			replacements := make([]*pb.OutboundPolicy_HeaderReplacement, 0, len(r.HeaderReplacements))
			for _, h := range r.HeaderReplacements {
				secretID := ""
				if h.Secret != nil {
					secretID = h.Secret.SecretID
				}
				replacements = append(replacements, pb.OutboundPolicy_HeaderReplacement_builder{
					SecretId: secretID,
					Headers:  maps.Clone(h.Headers),
				}.Build())
			}
			b.Https = pb.OutboundPolicy_HTTPSRoute_builder{
				Domains:            r.Domains,
				HeaderReplacements: replacements,
			}.Build()
		case OutboundRouteIP:
			b.Ip = pb.OutboundPolicy_IPRoute_builder{Cidrs: r.CIDRs}.Build()
		case OutboundRouteAny:
			var redirect *pb.OutboundPolicy_Redirect
			if blocks(r.Redirect) {
				redirect = pb.OutboundPolicy_Redirect_builder{
					Block: pb.OutboundPolicy_Redirect_Block_builder{}.Build(),
				}.Build()
			}
			b.Any = pb.OutboundPolicy_AnyRoute_builder{Redirect: redirect}.Build()
		}
		routes = append(routes, b.Build())
	}
	return pb.OutboundPolicy_builder{
		TlsDecryption: pb.OutboundPolicy_TLSDecryption_builder{Allowed: p.AllowTLSDecryption}.Build(),
		Routes:        routes,
	}.Build()
}
