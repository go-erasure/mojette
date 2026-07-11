//go:build amd64

package mojette

import (
	"bytes"
	"testing"
)

// TestForceKernelsAMD64 drives each amd64 kernel directly over whole blocks and
// finishes with the scalar tail, mirroring the dispatcher but without the size
// threshold, so every kernel is validated at every length. xorAVX2 runs only
// when the CPU has AVX2 (the VPXOR/VMOVDQU would #UD otherwise); xorSSE2 is the
// amd64 baseline and always runs.
func TestForceKernelsAMD64(t *testing.T) {
	force := func(k func(dst, src []byte) int) {
		for _, n := range xorSizes {
			dst := randBytes(n, int64(n)*3+1)
			src := randBytes(n, int64(n)*7+5)
			want := xorRef(dst, src)
			got := append([]byte(nil), dst...)
			done := k(got, src)
			xorBlockScalar(got[done:], src[done:])
			if !bytes.Equal(got, want) {
				t.Fatalf("n=%d: forced kernel mismatch", n)
			}
		}
	}
	force(xorSSE2)
	if hasAVX2 {
		force(xorAVX2)
	} else {
		t.Log("CPU lacks AVX2; xorAVX2 not exercised on this host")
	}
}

// TestDispatchAMD64 drives xorBlock down both the AVX2 and the SSE2 branch by
// toggling hasAVX2, restoring it with defer. With hasAVX2 forced off, every
// length >= 16 takes the SSE2 branch; with it on (the native CI runner), lengths
// >= 32 take AVX2 and 16..31 take SSE2 — covering all three switch arms.
func TestDispatchAMD64(t *testing.T) {
	saved := hasAVX2
	defer func() { hasAVX2 = saved }()

	run := func() {
		for _, n := range xorSizes {
			dst := randBytes(n, int64(n)*31+2)
			src := randBytes(n, int64(n)*37+4)
			want := xorRef(dst, src)
			got := append([]byte(nil), dst...)
			xorBlock(got, src)
			if !bytes.Equal(got, want) {
				t.Fatalf("hasAVX2=%v n=%d: xorBlock mismatch", hasAVX2, n)
			}
		}
	}

	hasAVX2 = false
	run()
	hasAVX2 = saved
	run()
	if !saved {
		t.Log("CPU lacks AVX2; AVX2 dispatch branch not exercised on this host")
	}
}
