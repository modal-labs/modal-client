package modal

import "time"

// idleCountdown counts down to giving something back once nothing is using it.
//
// The sequence number is what makes a callback that has already fired
// harmless. AfterFunc runs the callback on its own goroutine, so it can be
// waiting on the owner's lock while a new user arrives and arms a fresh
// countdown, and it must not then speak for the use that came after it.
//
// What counts as "in use", and what is given back, is the owner's business: one
// owner keeps a connection while operations are in flight, another keeps a
// stream while a read has it. This owns only the countdown, and the owner holds
// its own lock across every call here.
type idleCountdown struct {
	timer *time.Timer
	seq   uint64
}

// stop ends the countdown in progress, and retires a callback that has fired
// but not yet run. The sequence is bumped whether or not a timer is set, so a
// callback already past its own Stop finds a sequence it does not match.
func (i *idleCountdown) stop() {
	i.seq++
	if i.timer != nil {
		i.timer.Stop()
		i.timer = nil
	}
}

// arm starts the countdown again, calling release when it elapses. release
// runs on its own goroutine, without the owner's lock, and passes the sequence
// it was armed with back to fired. A timeout of zero or less never fires.
func (i *idleCountdown) arm(timeout time.Duration, release func(seq uint64)) {
	if timeout <= 0 {
		return
	}
	i.stop()
	seq := i.seq
	i.timer = time.AfterFunc(timeout, func() { release(seq) })
}

// fired reports whether seq belongs to the countdown in force. When it does,
// the timer has run and holds nothing more, so it is forgotten here.
func (i *idleCountdown) fired(seq uint64) bool {
	if seq != i.seq {
		return false
	}
	i.timer = nil
	return true
}
