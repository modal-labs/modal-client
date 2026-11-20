package modal

import (
	"context"
	"testing"

	pb "github.com/modal-labs/modal-client/go/proto/modal_proto"
	"github.com/onsi/gomega"
)

func TestSandboxCreateRequestProto_OutboundPolicy(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	secret := &Secret{SecretID: "st-123"}
	policy := &OutboundPolicy{
		AllowTLSDecryption: true,
		Routes: []OutboundRoute{
			OutboundRouteHTTPS{
				Domains: []string{"*.mydomain.com", "mydomain.org"},
				HeaderReplacements: []OutboundHeaderReplacement{
					{Secret: secret, Headers: map[string]string{"Authorization": "Bearer $API_TOKEN"}},
				},
			},
			&OutboundRouteHTTPS{Domains: []string{"api.partner.com"}},
			OutboundRouteTLS{Domains: []string{"*.example.com"}},
			OutboundRouteIP{CIDRs: []string{"203.0.113.0/24"}},
			OutboundRouteAny{Redirect: &OutboundBlock{}},
		},
	}

	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{OutboundPolicy: policy, Proxy: &Proxy{ProxyID: "pr-123"}})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())

	definition := req.GetDefinition()
	g.Expect(definition.HasExperimentalOutboundPolicy()).To(gomega.BeFalse())
	g.Expect(definition.GetNetworkAccess().GetNetworkAccessType()).To(gomega.Equal(pb.NetworkAccess_OPEN))
	got := definition.GetOutboundPolicy()
	g.Expect(got.GetTlsDecryption().GetAllowed()).To(gomega.BeTrue())
	routes := got.GetRoutes()
	g.Expect(routes).To(gomega.HaveLen(5))

	https := routes[0].GetHttps()
	g.Expect(https.GetDomains()).To(gomega.Equal([]string{"*.mydomain.com", "mydomain.org"}))
	g.Expect(https.GetHeaderReplacements()).To(gomega.HaveLen(1))
	g.Expect(https.GetHeaderReplacements()[0].GetSecretId()).To(gomega.Equal("st-123"))
	g.Expect(https.GetHeaderReplacements()[0].GetHeaders()).To(gomega.Equal(map[string]string{"Authorization": "Bearer $API_TOKEN"}))
	g.Expect(https.HasRedirect()).To(gomega.BeFalse())

	g.Expect(routes[1].GetHttps().GetDomains()).To(gomega.Equal([]string{"api.partner.com"}))
	g.Expect(routes[2].GetTls().GetDomains()).To(gomega.Equal([]string{"*.example.com"}))
	g.Expect(routes[3].GetIp().GetCidrs()).To(gomega.Equal([]string{"203.0.113.0/24"}))
	g.Expect(routes[3].GetIp().GetPorts()).To(gomega.BeEmpty())
	g.Expect(routes[3].GetIp().GetTransport()).To(gomega.Equal(pb.OutboundPolicy_TRANSPORT_UNSPECIFIED))
	g.Expect(routes[4].GetAny().GetRedirect().HasBlock()).To(gomega.BeTrue())

	g.Expect(policy.secrets()).To(gomega.Equal([]*Secret{secret}))
}

func TestSandboxCreateRequestProto_OutboundPolicyAllowAny(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	policy := &OutboundPolicy{Routes: []OutboundRoute{OutboundRouteAny{}}}
	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{OutboundPolicy: policy})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	got := req.GetDefinition().GetOutboundPolicy()
	g.Expect(got.GetTlsDecryption().GetAllowed()).To(gomega.BeFalse())
	g.Expect(got.GetRoutes()).To(gomega.HaveLen(1))
	g.Expect(got.GetRoutes()[0].HasAny()).To(gomega.BeTrue())
	g.Expect(got.GetRoutes()[0].GetAny().HasRedirect()).To(gomega.BeFalse())
}

func TestSandboxCreateRequestProto_NoOutboundPolicy(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	req, err := buildSandboxCreateRequestProto("app-123", "img-456", SandboxCreateParams{})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(req.GetDefinition().HasOutboundPolicy()).To(gomega.BeFalse())
}

func TestSandboxCreateRequestProto_OutboundPolicyInvalid(t *testing.T) {
	t.Parallel()

	blockAny := OutboundRouteAny{Redirect: OutboundBlock{}}
	withRoutes := func(allowTLSDecryption bool, routes ...OutboundRoute) *OutboundPolicy {
		return &OutboundPolicy{AllowTLSDecryption: allowTLSDecryption, Routes: append(routes, blockAny)}
	}
	https := func(headers map[string]string) OutboundRouteHTTPS {
		return OutboundRouteHTTPS{
			Domains:            []string{"api.example.com"},
			HeaderReplacements: []OutboundHeaderReplacement{{Headers: headers}},
		}
	}

	cases := []struct {
		name    string
		params  SandboxCreateParams
		message string
	}{
		{"no routes", SandboxCreateParams{OutboundPolicy: &OutboundPolicy{}}, "must end with an OutboundRouteAny"},
		{"no any route", SandboxCreateParams{OutboundPolicy: &OutboundPolicy{Routes: []OutboundRoute{
			OutboundRouteTLS{Domains: []string{"example.com"}},
		}}}, "must end with an OutboundRouteAny"},
		{"any route not last", SandboxCreateParams{OutboundPolicy: &OutboundPolicy{Routes: []OutboundRoute{
			blockAny, OutboundRouteTLS{Domains: []string{"example.com"}},
		}}}, "must be the last route"},
		{"two any routes", SandboxCreateParams{OutboundPolicy: &OutboundPolicy{Routes: []OutboundRoute{
			OutboundRouteAny{}, blockAny,
		}}}, "must be the last route"},
		{"nil route", SandboxCreateParams{OutboundPolicy: withRoutes(false, nil)}, "route is nil"},
		{"nil route pointer", SandboxCreateParams{OutboundPolicy: withRoutes(false, (*OutboundRouteTLS)(nil))}, "route is nil"},
		{"TLS route without domains", SandboxCreateParams{OutboundPolicy: withRoutes(false, OutboundRouteTLS{})}, "at least one domain"},
		{"invalid domain", SandboxCreateParams{OutboundPolicy: withRoutes(false, OutboundRouteTLS{Domains: []string{"not a domain"}})}, "invalid domain"},
		{"IP route without CIDRs", SandboxCreateParams{OutboundPolicy: withRoutes(false, OutboundRouteIP{})}, "at least one CIDR"},
		{"HTTPS route without TLS decryption", SandboxCreateParams{OutboundPolicy: withRoutes(false, OutboundRouteHTTPS{Domains: []string{"example.com"}})}, "require OutboundPolicy.AllowTLSDecryption"},
		{"no headers", SandboxCreateParams{OutboundPolicy: withRoutes(true, https(nil))}, "at least one header"},
		{"header set twice ignoring case", SandboxCreateParams{OutboundPolicy: withRoutes(true, https(map[string]string{"Authorization": "a", "authorization": "b"}))}, `header "authorization" is set more than once`},
		{"header set twice across replacements", SandboxCreateParams{OutboundPolicy: withRoutes(true, OutboundRouteHTTPS{
			Domains: []string{"example.com"},
			HeaderReplacements: []OutboundHeaderReplacement{
				{Headers: map[string]string{"X-Api-Key": "a"}},
				{Headers: map[string]string{"X-Api-Key": "b"}},
			},
		})}, `header "X-Api-Key" is set more than once`},
		{"template without secret", SandboxCreateParams{OutboundPolicy: withRoutes(true, https(map[string]string{"X-A": "$KEY"}))}, "no Secret was passed"},
		{"with BlockNetwork", SandboxCreateParams{OutboundPolicy: withRoutes(false), BlockNetwork: true}, "end Routes with OutboundRouteAny{Redirect: OutboundBlock{}}"},
		{"with CIDR allowlist", SandboxCreateParams{OutboundPolicy: withRoutes(false), OutboundCIDRAllowlist: &Allowlist{}}, "use an OutboundRouteIP"},
		{"with domain allowlist", SandboxCreateParams{OutboundPolicy: withRoutes(false), OutboundDomainAllowlist: &Allowlist{}}, "use an OutboundRouteTLS"},
		{"with experimental policy", SandboxCreateParams{OutboundPolicy: withRoutes(false), ExperimentalOutboundPolicy: &ExperimentalOutboundPolicy{}}, "use OutboundRouteHTTPS routes with HeaderReplacements"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			g := gomega.NewWithT(t)
			_, err := buildSandboxCreateRequestProto("app-123", "img-456", c.params)
			g.Expect(err).Should(gomega.MatchError(gomega.ContainSubstring(c.message)))
		})
	}
}

// failingModalClient implements no RPCs, so any control plane call panics.
type failingModalClient struct {
	pb.ModalClientClient
}

func TestSandboxCreateRejectsConflictingOutboundOptionsBeforeCallingModal(t *testing.T) {
	t.Parallel()

	for _, sandboxV2 := range []bool{false, true} {
		g := gomega.NewWithT(t)
		svc := &sandboxServiceImpl{client: &Client{
			cpClient: &clientWithConn{ModalClientClient: failingModalClient{}},
			profile:  Profile{SandboxV2: sandboxV2},
		}}

		// Getting past validation would build the image and turn the env vars
		// into a Secret. This client can do neither.
		_, err := svc.Create(context.Background(), &App{AppID: "ap-123"}, &Image{}, &SandboxCreateParams{
			Env:            map[string]string{"KEY": "value"},
			BlockNetwork:   true,
			OutboundPolicy: &OutboundPolicy{Routes: []OutboundRoute{OutboundRouteAny{}}},
		})
		g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("OutboundPolicy cannot be used when BlockNetwork is enabled")))
	}
}
