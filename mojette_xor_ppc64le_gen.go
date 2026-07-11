//go:build ignore

// Command gen produces mojette_xor_ppc64le.s with go-asmgen: the VSX kernel for
// the Mojette XOR region op dst[i] ^= src[i] over []byte.
//
// A VSX loop over whole 16-byte blocks. dst[i..] and src[i..] are loaded with
// LXVD2X into VS32 (=V0) and VS33 (=V1) — the VSX/VMX register-aliasing rule (as
// in go-simd/bitset): AltiVec Vn aliases VSX VS(32+n), so a load the VMX VXOR
// must see as V0/V1 targets VS32/VS33. VXOR writes V2 and STXVD2X stores VS34
// (=V2) back to dst. LXVD2X/STXVD2X are POWER8 (ISA 2.07) baseline, so no POWER9
// gate and no SIGILL on POWER8; because both load and store use the same
// doubleword-element move the element permutation cancels and XOR (bitwise) is
// unaffected. The Go wrapper (mojette_xor_ppc64le.go) finishes the 0..15-byte
// tail scalar-wise.
//
// Run: go run mojette_xor_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

func main() {
	f := emit.NewFile("ppc64le")

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src")},
		[]abi.Arg{abi.Scalar("done", abi.Int64)},
	)

	// R3=dst_base, R4=dst_len, R5=src_base, R6=src_len, R9=blocks, R10=offset.
	b := ppc64.NewFunc("xorVSX", sig, 0)
	b.LoadArg("dst_base", "R3").
		LoadArg("dst_len", "R4").
		LoadArg("src_base", "R5").
		LoadArg("src_len", "R6").
		// R4 = n = min(dst_len, src_len).
		Raw("CMP R6, R4").
		Raw("BGE usedst").
		Raw("MOVD R6, R4").
		Label("usedst").
		Raw("SRD $4, R4, R9").  // blocks = n >> 4
		Raw("MOVD $0, R10").    // byte offset / bytes consumed
		Label("loop").
		Raw("CMP R9, $0").
		Raw("BEQ done").
		Raw("ADD R5, R10, R12"). // &src[off]
		Raw("ADD R3, R10, R15"). // &dst[off]
		Raw("LXVD2X (R15)(R0), VS32"). // V0 = dst
		Raw("LXVD2X (R12)(R0), VS33"). // V1 = src
		Raw("VXOR V0, V1, V2").        // V2 = dst ^ src
		Raw("STXVD2X VS34, (R15)(R0)").
		Raw("ADD $16, R10, R10").
		Raw("ADD $-1, R9, R9").
		Raw("BR loop").
		Label("done").
		StoreRet("R10", "done").
		Ret()
	f.Add(b.Func())

	if err := os.WriteFile("mojette_xor_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote mojette_xor_ppc64le.s")
}
