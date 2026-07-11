//go:build ignore

// Command gen produces mojette_xor_amd64.s with go-asmgen: the amd64 SIMD
// kernels for the Mojette XOR region op dst[i] ^= src[i] over []byte.
//
// Two kernels, both an in-place vector XOR loop over whole blocks that read
// dst[i..] and src[i..], XOR them and store back to dst, returning the number of
// bytes consumed (a whole multiple of the block width):
//
//   - xorSSE2 over 16-byte blocks (MOVOU + PXOR): SSE2 is the amd64 baseline
//     (GOAMD64=v1), so it always runs, no feature flag.
//   - xorAVX2 over 32-byte blocks (VMOVDQU + VPXOR): dispatched only when the CPU
//     reports AVX2.
//
// XOR is bitwise, so the in-register byte order is irrelevant (load and store use
// the same unaligned move); the result is identical to the scalar reference. The
// Go wrapper (mojette_xor_amd64.go) finishes the 0..blk-1 tail scalar-wise.
//
// Run: go run mojette_xor_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func sig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src")},
		[]abi.Arg{abi.Scalar("done", abi.Int64)},
	)
}

// genSSE2 emits xorSSE2(dst, src []byte) (done int): 16 bytes/block via PXOR.
func genSSE2(f *emit.File) {
	b := amd64.NewFunc("xorSSE2", sig(), 0)
	b.LoadArg("dst_base", "DI").
		LoadArg("dst_len", "CX").
		LoadArg("src_base", "SI").
		LoadArg("src_len", "AX").
		Raw("CMPQ AX, CX"). // CX = n = min(dst_len, src_len)
		Raw("CMOVQLT AX, CX").
		Raw("MOVQ CX, R8"). // blocks = n >> 4
		Raw("SHRQ $4, R8").
		Raw("XORQ R9, R9"). // byte offset / bytes consumed
		Raw("TESTQ R8, R8").
		Raw("JZ done").
		Label("loop").
		Raw("MOVOU (DI)(R9*1), X0"). // dst
		Raw("MOVOU (SI)(R9*1), X1"). // src
		Raw("PXOR X1, X0").          // dst ^= src
		Raw("MOVOU X0, (DI)(R9*1)").
		Raw("ADDQ $16, R9").
		Raw("DECQ R8").
		Raw("JNZ loop").
		Label("done").
		StoreRet("R9", "done").
		Ret()
	f.Add(b.Func())
}

// genAVX2 emits xorAVX2(dst, src []byte) (done int): 32 bytes/block via VPXOR.
func genAVX2(f *emit.File) {
	b := amd64.NewFunc("xorAVX2", sig(), 0)
	b.LoadArg("dst_base", "DI").
		LoadArg("dst_len", "CX").
		LoadArg("src_base", "SI").
		LoadArg("src_len", "AX").
		Raw("CMPQ AX, CX"). // CX = n = min(dst_len, src_len)
		Raw("CMOVQLT AX, CX").
		Raw("MOVQ CX, R8"). // blocks = n >> 5
		Raw("SHRQ $5, R8").
		Raw("XORQ R9, R9"). // byte offset / bytes consumed
		Raw("TESTQ R8, R8").
		Raw("JZ done").
		Label("loop").
		Raw("VMOVDQU (DI)(R9*1), Y0"). // dst
		Raw("VMOVDQU (SI)(R9*1), Y1"). // src
		Raw("VPXOR Y1, Y0, Y0").       // dst ^= src
		Raw("VMOVDQU Y0, (DI)(R9*1)").
		Raw("ADDQ $32, R9").
		Raw("DECQ R8").
		Raw("JNZ loop").
		Label("done").
		Raw("VZEROUPPER").
		StoreRet("R9", "done").
		Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("amd64")
	genSSE2(f)
	genAVX2(f)
	if err := os.WriteFile("mojette_xor_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote mojette_xor_amd64.s")
}
