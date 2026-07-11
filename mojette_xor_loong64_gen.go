//go:build ignore

// Command gen produces mojette_xor_loong64.s with go-asmgen: the LSX kernel for
// the Mojette XOR region op dst[i] ^= src[i] over []byte.
//
// An LSX loop over whole 16-byte blocks: dst[i..] and src[i..] are loaded into
// 128-bit V-registers (VMOVQ), XORed with VXORV and stored back to dst (VMOVQ).
// XOR is bitwise so element ordering is irrelevant. LSX is treated as the loong64
// baseline (as in the sibling go-simd/bitset and go-simd/popcount), so there is
// no runtime feature check; the Go wrapper (mojette_xor_loong64.go) finishes the
// 0..15-byte tail scalar-wise.
//
// Run: go run mojette_xor_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
)

func main() {
	f := emit.NewFile("loong64")

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src")},
		[]abi.Arg{abi.Scalar("done", abi.Int64)},
	)

	// R4=dst_base, R5=dst_len, R6=src_base, R7=src_len, R10=blocks, R11=offset.
	b := loong64.NewFunc("xorLSX", sig, 0)
	b.LoadArg("dst_base", "R4").
		LoadArg("dst_len", "R5").
		LoadArg("src_base", "R6").
		LoadArg("src_len", "R7").
		// R5 = n = min(dst_len, src_len).
		Raw("BLT R7, R5, usesrc"). // src_len < dst_len -> n = src_len
		Raw("JMP haveN").
		Label("usesrc").
		Raw("MOVV R7, R5").
		Label("haveN").
		Raw("SRLV $4, R5, R10"). // blocks = n >> 4
		Raw("MOVV $0, R11").     // byte offset / bytes consumed
		Label("loop").
		Raw("BEQ R10, R0, done").
		Raw("ADDV R6, R11, R13"). // &src[off]
		Raw("ADDV R4, R11, R15"). // &dst[off]
		Raw("VMOVQ (R15), V0").   // dst
		Raw("VMOVQ (R13), V1").   // src
		Raw("VXORV V1, V0, V0").  // dst ^= src
		Raw("VMOVQ V0, (R15)").
		Raw("ADDV $16, R11, R11").
		Raw("ADDV $-1, R10, R10").
		Raw("JMP loop").
		Label("done").
		StoreRet("R11", "done").
		Ret()
	f.Add(b.Func())

	if err := os.WriteFile("mojette_xor_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote mojette_xor_loong64.s")
}
