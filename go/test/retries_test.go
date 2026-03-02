package test

import (
	"testing"
	"time"

	modal "github.com/modal-labs/modal-client/go"
	"github.com/onsi/gomega"
)

func TestRetriesConstructor(t *testing.T) {
	t.Parallel()
	g := gomega.NewWithT(t)

	backoff := float32(2.0)
	initial := 2 * time.Second
	max := 5 * time.Second
	r, err := modal.NewRetries(2, &modal.RetriesParams{
		BackoffCoefficient: &backoff,
		InitialDelay:       &initial,
		MaxDelay:           &max,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(r.MaxRetries).To(gomega.Equal(2))
	g.Expect(r.BackoffCoefficient).To(gomega.Equal(float32(2.0)))
	g.Expect(r.InitialDelay).To(gomega.Equal(2 * time.Second))
	g.Expect(r.MaxDelay).To(gomega.Equal(5 * time.Second))

	r, err = modal.NewRetries(3, nil)
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(r.MaxRetries).To(gomega.Equal(3))
	g.Expect(r.BackoffCoefficient).To(gomega.Equal(float32(2.0)))
	g.Expect(r.InitialDelay).To(gomega.Equal(1 * time.Second))
	g.Expect(r.MaxDelay).To(gomega.Equal(60 * time.Second))

	zeroDelay := 0 * time.Millisecond
	r, err = modal.NewRetries(1, &modal.RetriesParams{
		InitialDelay: &zeroDelay,
	})
	g.Expect(err).ShouldNot(gomega.HaveOccurred())
	g.Expect(r.InitialDelay).To(gomega.Equal(0 * time.Millisecond))
	g.Expect(r.BackoffCoefficient).To(gomega.Equal(float32(2.0))) // default
	g.Expect(r.MaxDelay).To(gomega.Equal(60 * time.Second))       // default

	_, err = modal.NewRetries(-1, nil)
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("maxRetries"))

	invalidBackoff := float32(0.9)
	_, err = modal.NewRetries(0, &modal.RetriesParams{BackoffCoefficient: &invalidBackoff})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("backoffCoefficient"))

	invalidInitial := 61 * time.Second
	_, err = modal.NewRetries(0, &modal.RetriesParams{InitialDelay: &invalidInitial})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("initialDelay"))

	invalidMax := 500 * time.Millisecond
	_, err = modal.NewRetries(0, &modal.RetriesParams{MaxDelay: &invalidMax})
	g.Expect(err).Should(gomega.HaveOccurred())
	g.Expect(err.Error()).Should(gomega.ContainSubstring("maxDelay"))
}
