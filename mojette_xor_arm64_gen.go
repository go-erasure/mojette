//go:build ignore

// Command gen produces mojette_xor_arm64.s with go-asmgen: the arm64 NEON kernel
// for the Mojette XOR region op dst[i] ^= src[i] over []byte.
//
// A NEON loop over whole 16-byte blocks: each iteration loads dst[i..i+15] and
// src[i..i+15] into V0/V1 (VLD1 .B16), applies VEOR (bitwise XOR, the .B16
// vector form is stable-Go assemblable) and stores V0 back to dst (VST1). It
// returns the number of bytes consumed (a multiple of 16); the Go wrapper
// (mojette_xor_arm64.go) finishes the 0..15-byte tail scalar-wise. NEON is the
// arm64 baseline, so there is no runtime feature check.
//
// Run: go run mojette_xor_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("arm64")

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src")},
		[]abi.Arg{abi.Scalar("done", abi.Int64)},
	)

	// R0=dst_base, R1=dst_len, R2=src_base, R3=src_len, R6=blocks, R7=offset.
	b := arm64.NewFunc("xorNEON", sig, 0)
	b.LoadArg("dst_base", "R0").
		LoadArg("dst_len", "R1").
		LoadArg("src_base", "R2").
		LoadArg("src_len", "R3").
		Raw("CMP R3, R1").              // n = min(dst_len, src_len)
		Raw("CSEL LT, R1, R3, R1").     // R1 = R1<R3 ? R1 : R3
		Raw("LSR $4, R1, R6").          // blocks = n >> 4
		Raw("MOVD $0, R7").             // byte offset / bytes consumed
		Label("loop").
		Raw("CBZ R6, done").
		Raw("ADD R0, R7, R10").         // &dst[off]
		Raw("ADD R2, R7, R9").          // &src[off]
		Raw("VLD1 (R10), [V0.B16]").    // dst
		Raw("VLD1 (R9), [V1.B16]").     // src
		Raw("VEOR V1.B16, V0.B16, V0.B16"). // dst ^= src
		Raw("VST1 [V0.B16], (R10)").
		Raw("ADD $16, R7, R7").
		Raw("SUB $1, R6, R6").
		Raw("JMP loop").
		Label("done").
		StoreRet("R7", "done").
		Ret()
	f.Add(b.Func())

	if err := os.WriteFile("mojette_xor_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote mojette_xor_arm64.s")
}
