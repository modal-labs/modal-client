package modal

import (
	"fmt"
	"testing"

	"github.com/onsi/gomega"
)

func TestTemplateReferencesKey(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	cases := []struct {
		template string
		want     bool
	}{
		{"Bearer $API_KEY", true},
		{"$_PRIVATE", true},
		{"x$y", true},
		{"plain", false},
		{"ca$$h", false},
		{"$$API_KEY", false},
		{"$5", false},
		{"$", false},
		{"ends with $", false},
		{"ca$$h $KEY", true},
	}
	for _, c := range cases {
		g.Expect(templateReferencesKey(c.template)).To(gomega.Equal(c.want), "template %q", c.template)
	}
}

func TestWithHeaderReplacementBuildsImmutablePolicy(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	var policy ExperimentalOutboundPolicy
	updated := policy.WithHeaderReplacement(ExperimentalOutboundPolicyHeaderReplacement{
		Domain:  "api.example.com",
		Headers: map[string]string{"X-Static": "plain"},
	})
	g.Expect(policy.replacements).To(gomega.BeEmpty())
	g.Expect(updated.replacements).To(gomega.HaveLen(1))
	g.Expect(updated.replacements[0].Domain).To(gomega.Equal("api.example.com"))
	g.Expect(updated.replacements[0].Headers).To(gomega.Equal(map[string]string{"X-Static": "plain"}))
	g.Expect(updated.replacements[0].Secret).To(gomega.BeNil())
}

func TestNewExperimentalOutboundPolicyMatchesChainedWith(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	replacements := []ExperimentalOutboundPolicyHeaderReplacement{
		{Domain: "api.example.com", Headers: map[string]string{"Authorization": "Bearer $API_KEY"}},
		{Domain: "*.example.com", Headers: map[string]string{"X-Static": "plain"}},
	}
	declarative := NewExperimentalOutboundPolicy(replacements...)
	imperative := new(ExperimentalOutboundPolicy).
		WithHeaderReplacement(replacements[0]).
		WithHeaderReplacement(replacements[1])
	g.Expect(declarative.replacements).To(gomega.Equal(imperative.replacements))
}

func TestExperimentalOutboundPolicyToProtoFansOutReplacements(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	secret := &Secret{SecretID: "st-1"}
	policy := NewExperimentalOutboundPolicy(
		ExperimentalOutboundPolicyHeaderReplacement{
			Domain:  "api.example.com",
			Secret:  secret,
			Headers: map[string]string{"Authorization": "Bearer $API_KEY", "X-Key-Raw": "$API_KEY"},
		},
		ExperimentalOutboundPolicyHeaderReplacement{
			Domain:  "*.example.com",
			Headers: map[string]string{"X-Static": "plain"},
		},
	)

	replacements := policy.toProto().GetHeaderReplacements()
	g.Expect(replacements).To(gomega.HaveLen(2))
	g.Expect(replacements[0].GetDomain()).To(gomega.Equal("api.example.com"))
	g.Expect(replacements[0].GetSecretId()).To(gomega.Equal("st-1"))
	g.Expect(replacements[0].GetHeaders()).To(gomega.Equal(map[string]string{
		"Authorization": "Bearer $API_KEY",
		"X-Key-Raw":     "$API_KEY",
	}))
	g.Expect(replacements[1].GetDomain()).To(gomega.Equal("*.example.com"))
	g.Expect(replacements[1].GetSecretId()).To(gomega.Equal(""))
	g.Expect(replacements[1].GetHeaders()).To(gomega.Equal(map[string]string{"X-Static": "plain"}))
}

// Builders perform no validation; invalid replacements are rejected by
// validate(), called when the policy is used.
func TestExperimentalOutboundPolicyValidate(t *testing.T) {
	t.Parallel()

	validate := func(r ExperimentalOutboundPolicyHeaderReplacement) error {
		p := NewExperimentalOutboundPolicy(r)
		return p.validate()
	}

	t.Run("invalid domain", func(t *testing.T) {
		t.Parallel()
		g := gomega.NewWithT(t)
		for _, domain := range []string{"not a domain", "example.com\n"} {
			err := validate(ExperimentalOutboundPolicyHeaderReplacement{
				Domain:  domain,
				Headers: map[string]string{"a": "b"},
			})
			g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("Invalid domain")), "domain %q", domain)
		}
	})

	t.Run("wildcards accepted", func(t *testing.T) {
		t.Parallel()
		g := gomega.NewWithT(t)
		policy := NewExperimentalOutboundPolicy(
			ExperimentalOutboundPolicyHeaderReplacement{Domain: "*.example.com", Headers: map[string]string{"a": "b"}},
			ExperimentalOutboundPolicyHeaderReplacement{Domain: "*", Headers: map[string]string{"a": "b"}},
		)
		g.Expect(policy.validate()).ToNot(gomega.HaveOccurred())
	})

	t.Run("empty headers", func(t *testing.T) {
		t.Parallel()
		g := gomega.NewWithT(t)
		err := validate(ExperimentalOutboundPolicyHeaderReplacement{
			Domain:  "example.com",
			Headers: map[string]string{},
		})
		g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("at least one header")))
	})

	t.Run("invalid header name", func(t *testing.T) {
		t.Parallel()
		g := gomega.NewWithT(t)
		for _, name := range []string{"bad header", "X-Foo\n"} {
			err := validate(ExperimentalOutboundPolicyHeaderReplacement{
				Domain:  "example.com",
				Headers: map[string]string{name: "x"},
			})
			g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("Invalid header name")), "name %q", name)
		}
	})

	t.Run("control characters in header value", func(t *testing.T) {
		t.Parallel()
		g := gomega.NewWithT(t)
		for _, value := range []string{"x\r\ninjected", "x\x00y", "x\x7fy"} {
			err := validate(ExperimentalOutboundPolicyHeaderReplacement{
				Domain:  "example.com",
				Headers: map[string]string{"a": value},
			})
			g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("control characters")), "value %q", value)
		}
	})

	t.Run("tab in header value", func(t *testing.T) {
		t.Parallel()
		g := gomega.NewWithT(t)
		err := validate(ExperimentalOutboundPolicyHeaderReplacement{
			Domain:  "example.com",
			Headers: map[string]string{"a": "x\ty"},
		})
		g.Expect(err).ToNot(gomega.HaveOccurred())
	})

	t.Run("template without secret", func(t *testing.T) {
		t.Parallel()
		g := gomega.NewWithT(t)
		err := validate(ExperimentalOutboundPolicyHeaderReplacement{
			Domain:  "example.com",
			Headers: map[string]string{"Authorization": "Bearer $API_KEY"},
		})
		g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("no Secret was passed")))
	})

	t.Run("escaped dollar without secret", func(t *testing.T) {
		t.Parallel()
		g := gomega.NewWithT(t)
		err := validate(ExperimentalOutboundPolicyHeaderReplacement{
			Domain:  "example.com",
			Headers: map[string]string{"a": "ca$$h", "b": "$5", "c": "$"},
		})
		g.Expect(err).ToNot(gomega.HaveOccurred())
	})

	t.Run("ephemeral secret", func(t *testing.T) {
		t.Parallel()
		g := gomega.NewWithT(t)
		ephemeral := &Secret{hydrator: &secretFromMapHydrator{envDict: map[string]string{"API_KEY": "k"}}}
		err := validate(ExperimentalOutboundPolicyHeaderReplacement{
			Domain:  "example.com",
			Secret:  ephemeral,
			Headers: map[string]string{"Authorization": "Bearer $API_KEY"},
		})
		g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("only support named secrets")))
	})

	t.Run("too many headers", func(t *testing.T) {
		t.Parallel()
		g := gomega.NewWithT(t)
		headers := map[string]string{}
		for i := range maxOutboundPolicyHeaderReplacements + 1 {
			headers[string(rune('a'+i))] = "x"
		}
		err := validate(ExperimentalOutboundPolicyHeaderReplacement{
			Domain:  "example.com",
			Headers: headers,
		})
		g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("more than 25 header replacements")))

		// The limit counts headers across replacements.
		full := map[string]string{}
		for i := range maxOutboundPolicyHeaderReplacements {
			full[fmt.Sprintf("X-Header-%d", i)] = "x"
		}
		policy := NewExperimentalOutboundPolicy(
			ExperimentalOutboundPolicyHeaderReplacement{Domain: "example.com", Headers: map[string]string{"a": "b"}},
			ExperimentalOutboundPolicyHeaderReplacement{Domain: "other.com", Headers: full},
		)
		err = policy.validate()
		g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring("more than 25 header replacements")))
	})

	t.Run("nil policy is valid", func(t *testing.T) {
		t.Parallel()
		g := gomega.NewWithT(t)
		var empty *ExperimentalOutboundPolicy
		g.Expect(empty.validate()).ToNot(gomega.HaveOccurred())
	})
}

func TestExperimentalOutboundPolicySecretsDeduplicates(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	secret := &Secret{SecretID: "st-1"}
	policy := NewExperimentalOutboundPolicy(
		ExperimentalOutboundPolicyHeaderReplacement{Domain: "a.example.com", Secret: secret, Headers: map[string]string{"a": "$K"}},
		ExperimentalOutboundPolicyHeaderReplacement{Domain: "b.example.com", Secret: secret, Headers: map[string]string{"b": "$K"}},
	)
	g.Expect(policy.secrets()).To(gomega.Equal([]*Secret{secret}))

	var empty *ExperimentalOutboundPolicy
	g.Expect(empty.secrets()).To(gomega.BeEmpty())
	g.Expect(empty.toProto()).To(gomega.BeNil())
}
