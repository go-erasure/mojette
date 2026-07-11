//go:build ignore

// Command gen produces mojette_xor_s390x.s with go-asmgen: the vector-facility
// kernel for the Mojette XOR region op dst[i] ^= src[i] over []byte.
//
// A vector loop over whole 16-byte blocks: dst[i..] and src[i..] are loaded with
// VL into V0,V1, combined with VX (3-operand, dst last: "VX V1, V0, V2" means
// V2 = V0 ^ V1) and stored back to dst with VST. s390x is big-endian, but VX is a
// bitwise per-element XOR that does not depend on lane numbering, so loading,
// XORing and storing with the same VL/VST is endian-neutral and equal to the
// scalar reference. The vector facility (z13) is the s390x baseline, so no
// runtime feature flag; the Go wrapper (mojette_xor_s390x.go) finishes the
// 0..15-byte tail scalar-wise.
//
// Run: go run mojette_xor_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func main() {
	f := emit.NewFile("s390x")

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src")},
		[]abi.Arg{abi.Scalar("done", abi.Int64)},
	)

	// R1=dst_base, R2=dst_len, R3=src_base, R4=src_len, R7=blocks, R8=offset.
	b := s390x.NewFunc("xorVX", sig, 0)
	b.LoadArg("dst_base", "R1").
		LoadArg("dst_len", "R2").
		LoadArg("src_base", "R3").
		LoadArg("src_len", "R4").
		// R2 = n = min(dst_len, src_len).
		Raw("CMPBLE R2, R4, haveN"). // dst_len <= src_len -> n = dst_len
		Raw("MOVD R4, R2").
		Label("haveN").
		Raw("SRD $4, R2, R7"). // blocks = n >> 4
		Raw("MOVD $0, R8").    // byte offset / bytes consumed
		Label("loop").
		Raw("CMPBEQ R7, $0, done").
		Raw("ADD R3, R8, R11"). // &src[off]
		Raw("ADD R1, R8, R12"). // &dst[off]
		Raw("VL (R12), V0").    // dst
		Raw("VL (R11), V1").    // src
		Raw("VX V1, V0, V2").   // V2 = dst ^ src
		Raw("VST V2, (R12)").
		Raw("ADD $16, R8, R8").
		Raw("ADD $-1, R7, R7").
		Raw("BR loop").
		Label("done").
		StoreRet("R8", "done").
		Ret()
	f.Add(b.Func())

	if err := os.WriteFile("mojette_xor_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote mojette_xor_s390x.s")
}
