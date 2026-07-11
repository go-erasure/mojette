//go:build riscv64

package mojette

import (
	"bytes"
	"testing"
)

// TestDispatchRISCV64 drives both riscv64 branches. With the V extension present
// (the QEMU rv64,v=true target) xorBlock folds the whole region through the RVV
// kernel; forcing hasV low exercises the scalar fallback. Both are checked
// byte-for-byte against the oracle so a single host reaches 100% coverage.
func TestDispatchRISCV64(t *testing.T) {
	saved := hasV
	defer func() { hasV = saved }()

	run := func(tag string) {
		for _, n := range xorSizes {
			dst := randBytes(n, int64(n)*31+2)
			src := randBytes(n, int64(n)*37+4)
			want := xorRef(dst, src)
			got := append([]byte(nil), dst...)
			xorBlock(got, src)
			if !bytes.Equal(got, want) {
				t.Fatalf("%s n=%d: xorBlock mismatch", tag, n)
			}
		}
	}

	run("hasV") // real flag: RVV kernel under v=true
	hasV = false
	run("noV") // forced scalar fallback
	if !saved {
		t.Log("CPU lacks V; RVV kernel not exercised on this host")
	}
}
