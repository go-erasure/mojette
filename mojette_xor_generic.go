//go:build !amd64 && !arm64 && !riscv64 && !ppc64le && !s390x && !loong64

package mojette

// No SIMD kernel on this architecture, so the Mojette XOR region op uses the
// portable scalar loop.
func xorBlock(dst, src []byte) { xorBlockScalar(dst, src) }
