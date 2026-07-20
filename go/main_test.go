package modal

import (
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	err := os.Setenv("MODAL_ENVIRONMENT", "libmodal")
	if err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}
