package mojette

import (
	"bytes"
	"math/rand"
	"testing"
)

// xorRef is an independent oracle for dst[i] ^= src[i], deliberately distinct
// from the package's xorBlockScalar, so the differential test checks the SIMD
// dispatch against a second source of truth.
func xorRef(dst, src []byte) []byte {
	out := append([]byte(nil), dst...)
	n := len(out)
	if len(src) < n {
		n = len(src)
	}
	for i := 0; i < n; i++ {
		out[i] ^= src[i]
	}
	return out
}

// xorSizes sweeps lengths across the SIMD-block / tail boundary: empty, 1 byte,
// sub-block, exact multiples of 16/32, and non-multiples, plus large buffers.
var xorSizes = []int{
	0, 1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65,
	100, 127, 128, 255, 256, 257, 1000, 4096, 4097, 65536, 65537,
}

func randBytes(n int, seed int64) []byte {
	b := make([]byte, n)
	rand.New(rand.NewSource(seed)).Read(b)
	return b
}

// TestXorBlockDifferential is the correctness oracle: for every length it XORs
// random dst/src through xorBlock (the SIMD dispatch on this arch) and asserts
// byte-identical output to the independent scalar oracle.
func TestXorBlockDifferential(t *testing.T) {
	for _, n := range xorSizes {
		dst := randBytes(n, int64(n)*3+1)
		src := randBytes(n, int64(n)*7+5)
		want := xorRef(dst, src)
		got := append([]byte(nil), dst...)
		xorBlock(got, src)
		if !bytes.Equal(got, want) {
			t.Fatalf("n=%d: xorBlock mismatch\n got=%x\nwant=%x", n, got, want)
		}
	}
}

// TestXorBlockMismatched exercises the min-length guard (src shorter than dst):
// only the first len(src) bytes of dst are touched, the rest left unchanged.
// This covers the `n = len(src)` statement in both xorBlock and xorBlockScalar.
func TestXorBlockMismatched(t *testing.T) {
	for _, n := range xorSizes {
		if n == 0 {
			continue
		}
		short := n / 2
		dst := randBytes(n, int64(n)*23+1)
		src := randBytes(short, int64(n)*29+7)

		want := xorRef(dst, src)
		got := append([]byte(nil), dst...)
		xorBlock(got, src)
		if !bytes.Equal(got, want) {
			t.Fatalf("n=%d short=%d: xorBlock mismatch", n, short)
		}

		want = xorRef(dst, src)
		gots := append([]byte(nil), dst...)
		xorBlockScalar(gots, src)
		if !bytes.Equal(gots, want) {
			t.Fatalf("n=%d short=%d: xorBlockScalar mismatch", n, short)
		}
	}
}

// TestXorBlockScalarOracle checks the package scalar reference (the tail and the
// no-SIMD fallback) against the independent oracle so its coverage is genuine
// even on a SIMD host.
func TestXorBlockScalarOracle(t *testing.T) {
	for _, n := range xorSizes {
		dst := randBytes(n, int64(n)*11+2)
		src := randBytes(n, int64(n)*13+4)
		want := xorRef(dst, src)
		got := append([]byte(nil), dst...)
		xorBlockScalar(got, src)
		if !bytes.Equal(got, want) {
			t.Fatalf("n=%d: xorBlockScalar mismatch", n)
		}
	}
}

// TestXorBlockPatterns exercises the identities XOR must satisfy (self -> zero,
// zero -> identity, involution) across the block boundaries.
func TestXorBlockPatterns(t *testing.T) {
	for _, n := range xorSizes {
		a := randBytes(n, int64(n)*17+9)

		// a ^ a == 0.
		self := append([]byte(nil), a...)
		xorBlock(self, a)
		for i, v := range self {
			if v != 0 {
				t.Fatalf("n=%d a^a byte %d = %#x, want 0", n, i, v)
			}
		}

		// a ^ 0 == a.
		id := append([]byte(nil), a...)
		xorBlock(id, make([]byte, n))
		if !bytes.Equal(id, a) {
			t.Fatalf("n=%d a^0 != a", n)
		}

		// involution: (a ^ b) ^ b == a.
		b := randBytes(n, int64(n)*19+3)
		inv := append([]byte(nil), a...)
		xorBlock(inv, b)
		xorBlock(inv, b)
		if !bytes.Equal(inv, a) {
			t.Fatalf("n=%d (a^b)^b != a", n)
		}
	}
}

// TestEncodeReconstructSIMDLargeBlock drives the public API through xorBlock with
// a block size well past the vector width, so Encode/Reconstruct exercise the
// SIMD fast path end to end and still round-trip exactly.
func TestEncodeReconstructSIMDLargeBlock(t *testing.T) {
	const blockSize = 4096 + 7 // past AVX2 width, with a tail
	g := makeGrid(4, 3, blockSize)
	dirs := []Direction{{-1, 1}, {0, 1}, {1, 1}, {2, 1}}
	projs, err := Encode(g, dirs)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	subset := []Projection{projs[0], projs[2], projs[3]} // sum|P| = 4 >= cols
	got, err := Reconstruct(4, 3, blockSize, subset)
	if err != nil {
		t.Fatalf("reconstruct: %v", err)
	}
	if !gridsEqual(got, g) {
		t.Fatal("SIMD round-trip mismatch")
	}
}

// FuzzXorBlock is the differential fuzzer: it XORs two arbitrary fuzzer-chosen
// buffers through xorBlock (the SIMD dispatch) and the independent scalar oracle
// and asserts byte-identical output. The two inputs need not be equal length —
// the min-length guard is fuzzed too.
func FuzzXorBlock(f *testing.F) {
	f.Add([]byte{1, 2, 3}, []byte{4, 5, 6})
	f.Add([]byte(nil), []byte(nil))
	f.Add(bytes.Repeat([]byte{0xa5}, 100), bytes.Repeat([]byte{0x5a}, 100))
	f.Fuzz(func(t *testing.T, dst, src []byte) {
		want := xorRef(dst, src)
		got := append([]byte(nil), dst...)
		xorBlock(got, src)
		if !bytes.Equal(got, want) {
			t.Fatalf("dst=%d src=%d: xorBlock != oracle", len(dst), len(src))
		}
	})
}

func benchXor(b *testing.B, fn func(dst, src []byte)) {
	src := randBytes(1<<20, 1)
	dst := randBytes(1<<20, 2)
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(dst, src)
	}
}

// BenchmarkXorBlockScalar and BenchmarkXorBlockSIMD compare the scalar loop with
// the dispatched SIMD kernel on a 1 MiB buffer.
func BenchmarkXorBlockScalar(b *testing.B) { benchXor(b, xorBlockScalar) }
func BenchmarkXorBlockSIMD(b *testing.B)   { benchXor(b, xorBlock) }
