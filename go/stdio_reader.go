package modal

import (
	"context"
	"sync"
)

// chunkReader is the half the two stdio readers share: Read hands over what the
// last fetch produced, and fetches again only once that runs out.
//
// What differs between them - which RPC to open, how to retry it, and how a
// message maps onto bytes - stays with the reader that embeds this and reaches
// it through fetch and dropStream.
type chunkReader struct {
	ctx    context.Context
	cancel context.CancelFunc

	mu      sync.Mutex
	pending []byte
	err     error

	// fetch puts the next piece of output into pending. Called under mu, and
	// only with pending empty.
	fetch func() error
	// dropStream forgets whatever fetch opened, so the next fetch reopens it.
	// Called under mu.
	dropStream func()
	// inUse and idle, when set, bracket the part of Read where the caller is
	// here wanting output. Both are called under mu, so a reader that keeps
	// state about being read from can touch it there.
	inUse func()
	idle  func()
}

func (r *chunkReader) Read(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// A read that asks for nothing waits for nothing: io.Reader gives a
	// zero-length read no reason to go looking for output. It still reports an
	// error already known, which the contract allows.
	if len(p) == 0 {
		return 0, r.err
	}

	if r.inUse != nil {
		r.inUse()
		defer r.idle()
	}

	for len(r.pending) == 0 {
		if r.err != nil {
			return 0, r.err
		}
		if err := r.fetch(); err != nil {
			r.err = err
			r.dropStream()
			return 0, err
		}
	}

	n := copy(p, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

// Close ends the read and the stream under it.
//
// The cancel comes before the lock on purpose: a Read blocked on the stream
// holds the lock until it returns, and cancelling is what makes it return.
func (r *chunkReader) Close() error {
	r.cancel()

	r.mu.Lock()
	defer r.mu.Unlock()
	r.dropStream()
	return nil
}
