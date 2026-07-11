//go:build ignore

// Command gen produces mojette_xor_riscv64.s with go-asmgen: an RVV (vector)
// kernel for the Mojette XOR region op dst[i] ^= src[i] over []byte.
//
// A length-agnostic RVV loop: VSETVLI grants a strip length vl (SEW=8, LMUL=1)
// each iteration, the strip of dst and src bytes is loaded (VLE8V), XORed
// (VXORVV) and stored back to dst (VSE8V). The loop consumes the whole region
// (min of the two slice lengths) so no scalar tail remains when the kernel runs;
// it returns the byte count consumed. The V (vector) instructions trap on a hart
// without the extension, so the Go wrapper (mojette_xor_riscv64.go) only
// dispatches here when cpu.RISCV64.HasV is set, else it uses the scalar loop.
//
// Run: go run mojette_xor_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func main() {
	f := emit.NewFile("riscv64")

	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src")},
		[]abi.Arg{abi.Scalar("done", abi.Int64)},
	)

	// X5=dst_base, X6=dst_len, X7=src_base, X8=src_len, X9=vl, X10=done.
	b := riscv64.NewFunc("xorRVV", sig, 0)
	b.LoadArg("dst_base", "X5").
		LoadArg("dst_len", "X6").
		LoadArg("src_base", "X7").
		LoadArg("src_len", "X8").
		// X6 = n = min(dst_len, src_len).
		Raw("BGE X6, X8, usesrc"). // dst_len >= src_len -> n = src_len
		Raw("JMP haveN").
		Label("usesrc").
		Raw("MOV X8, X6").
		Label("haveN").
		Raw("MOV $0, X10"). // done
		Label("loop").
		Raw("BEQZ X6, done").
		Raw("VSETVLI X6, E8, M1, TA, MA, X9"). // X9 = granted vl
		Raw("VLE8V (X5), V0").                 // dst strip
		Raw("VLE8V (X7), V1").                 // src strip
		Raw("VXORVV V1, V0, V0").              // dst ^= src
		Raw("VSE8V V0, (X5)").
		Raw("ADD X9, X5, X5").
		Raw("ADD X9, X7, X7").
		Raw("SUB X9, X6, X6").
		Raw("ADD X9, X10, X10").
		Raw("JMP loop").
		Label("done").
		StoreRet("X10", "done").
		Ret()
	f.Add(b.Func())

	if err := os.WriteFile("mojette_xor_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote mojette_xor_riscv64.s")
}
