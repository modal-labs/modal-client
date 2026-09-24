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

const maxOutboundPolicyHeaderReplacements = 25

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

// ExperimentalOutboundPolicyHeaderReplacement is a single domain-scoped group of header
// replacements.
type ExperimentalOutboundPolicyHeaderReplacement struct {
	// Domain the replacements are scoped to. Supports "*." wildcard prefixes
	// (matching the apex domain and subdomains) and a bare "*".
	Domain string
	// Headers maps header names to values. Values support `$KEY` templates
	// referencing keys in the replacement's Secret. Literal `$` characters are
	// written `$$`.
	Headers map[string]string
	// Secret is a named Secret (e.g. from Client.Secrets.FromName) whose keys
	// may be referenced in the header value templates. Static replacements pass nil.
	Secret *Secret
}

// ExperimentalOutboundPolicy is an immutable configuration for replacing headers in
// outbound HTTPS requests from a Sandbox.
//
// EXPERIMENTAL: the API is subject to change.
//
// Secret values never enter the Sandbox: they are resolved and injected into
// matching requests outside the container.
//
//	secret, _ := client.Secrets.FromName(ctx, "api-token", nil)
//	outboundPolicy := modal.NewExperimentalOutboundPolicy(
//		modal.ExperimentalOutboundPolicyHeaderReplacement{
//			Domain:  "example.com",
//			Secret:  secret,
//			Headers: map[string]string{"Authorization": "Bearer $API_TOKEN"},
//		},
//		modal.ExperimentalOutboundPolicyHeaderReplacement{
//			Domain:  "modal.com",
//			Headers: map[string]string{"X-Trace-Token": "trace_abcd"},
//		},
//	)
type ExperimentalOutboundPolicy struct {
	replacements []ExperimentalOutboundPolicyHeaderReplacement
}

// NewExperimentalOutboundPolicy returns an ExperimentalOutboundPolicy with the given header replacements.
func NewExperimentalOutboundPolicy(replacements ...ExperimentalOutboundPolicyHeaderReplacement) ExperimentalOutboundPolicy {
	p := ExperimentalOutboundPolicy{}
	for _, r := range replacements {
		p = p.WithHeaderReplacement(r)
	}
	return p
}

// WithHeaderReplacement returns a new ExperimentalOutboundPolicy with an added header replacement.
func (p ExperimentalOutboundPolicy) WithHeaderReplacement(replacement ExperimentalOutboundPolicyHeaderReplacement) ExperimentalOutboundPolicy {
	headers := make(map[string]string, len(replacement.Headers))
	maps.Copy(headers, replacement.Headers)
	replacement.Headers = headers
	return ExperimentalOutboundPolicy{replacements: append(slices.Clone(p.replacements), replacement)}
}

// validate checks all replacements, returning the first violation found.
func (p *ExperimentalOutboundPolicy) validate() error {
	if p == nil {
		return nil
	}
	total := 0
	for _, r := range p.replacements {
		if r.Domain != "*" && !outboundPolicyDomainRe.MatchString(r.Domain) {
			return InvalidError{Exception: fmt.Sprintf("Invalid domain: %q", r.Domain)}
		}
		if len(r.Headers) == 0 {
			return InvalidError{Exception: "Headers must contain at least one header"}
		}
		// Sort the names for deterministic error messages and proto conversion.
		for _, name := range slices.Sorted(maps.Keys(r.Headers)) {
			value := r.Headers[name]
			if !outboundPolicyHeaderNameRe.MatchString(name) {
				return InvalidError{Exception: fmt.Sprintf("Invalid header name: %q", name)}
			}
			if hasInvalidHeaderValueChar(value) {
				return InvalidError{Exception: fmt.Sprintf("Header value for %q must not contain control characters (except tab)", name)}
			}
			if r.Secret == nil && templateReferencesKey(value) {
				return InvalidError{Exception: fmt.Sprintf(
					"Header value for %q references a secret key, but no Secret was passed", name)}
			}
		}
		if r.Secret != nil {
			if _, isEnvDict := secretEnvDictHydrator(r.Secret); isEnvDict {
				return InvalidError{Exception: "Outbound policies only support named secrets" +
					" (e.g. Client.Secrets.FromName); ephemeral secrets (e.g. Client.Secrets.FromMap)" +
					" are not yet supported"}
			}
		}
		total += len(r.Headers)
	}
	if total > maxOutboundPolicyHeaderReplacements {
		return InvalidError{Exception: fmt.Sprintf(
			"Outbound policy cannot have more than %d header replacements", maxOutboundPolicyHeaderReplacements)}
	}
	return nil
}

// secrets returns the deduplicated Secrets referenced by the policy's replacements.
func (p *ExperimentalOutboundPolicy) secrets() []*Secret {
	if p == nil {
		return nil
	}
	var out []*Secret
	for _, replacement := range p.replacements {
		if replacement.Secret != nil && !slices.Contains(out, replacement.Secret) {
			out = append(out, replacement.Secret)
		}
	}
	return out
}

// toProto converts to the wire format. Referenced secrets must be hydrated first.
func (p *ExperimentalOutboundPolicy) toProto() *pb.OutboundPolicy {
	if p == nil {
		return nil
	}
	replacements := make([]*pb.OutboundPolicy_HeaderReplacement, 0, len(p.replacements))
	for _, replacement := range p.replacements {
		secretID := ""
		if replacement.Secret != nil {
			secretID = replacement.Secret.SecretID
		}
		replacements = append(replacements, pb.OutboundPolicy_HeaderReplacement_builder{
			Domain:   replacement.Domain,
			SecretId: secretID,
			Headers:  replacement.Headers,
		}.Build())
	}
	return pb.OutboundPolicy_builder{HeaderReplacements: replacements}.Build()
}
