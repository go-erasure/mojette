# mojette

[![ci](https://github.com/go-erasure/mojette/actions/workflows/ci.yml/badge.svg)](https://github.com/go-erasure/mojette/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/go-erasure/mojette.svg)](https://pkg.go.dev/github.com/go-erasure/mojette)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

Pure-Go implementation of the **Mojette transform** erasure code: the discrete
Radon-projection code used by [RozoFS](https://rozofs.github.io/rozofs/master/),
operating **XOR-only** (GF(2) addition) over fixed-size byte blocks.

- **CGO_ENABLED=0**, stdlib only, **zero third-party dependencies**.
- Go 1.26.4 floor; builds for `linux/{amd64,arm64,riscv64,ppc64le,s390x,loong64}`,
  `darwin/{amd64,arm64}`, and `windows/amd64`.
- 100% test coverage.

## Model

Data is a **Grid** of `Rows`×`Cols` blocks in row-major order, each block
`BlockSize` bytes. A **Direction** `(P, Q)` (with `Q >= 1` and `gcd(|P|, Q) == 1`)
defines a family of discrete lines. Its **Projection** has one bin block per line;
each grid block is XOR-accumulated into the bin its cell falls on.

For cell `(i, j)` (row `i ∈ [0, Rows)`, col `j ∈ [0, Cols)`) and direction `(P, Q)`:

```
raw    = Q*i - P*j
minRaw = (P > 0) ? -P*(Cols-1) : 0
bin    = raw - minRaw                       // >= 0
nbins  = Q*(Rows-1) + |P|*(Cols-1) + 1
```

`Encode` does `bins[bin(i,j)] ^= data[i*Cols+j]` for every cell.

Reconstruction is the iterative inverse Mojette (corner / back-projection):
repeatedly find a bin whose contributing cells are all known but one, assign that
cell the bin's current value, XOR it back out of every projection, and repeat.

Because it is **XOR-only**, this is an erasure code for whole blocks (recover
missing projections), not a general error-correcting code.

### Katz criterion

`Reconstructible(rows, cols, dirs)` implements the Katz sufficiency criterion:

```
sum(Q_k) >= rows   OR   sum(|P_k|) >= cols
```

Katz is a *sufficient* geometric bound in the ideal Mojette setting; a particular
subset of projections is guaranteed reconstructible when it holds. `Reconstruct`
returns `ErrNotReconstructible` when the supplied projections do not actually
suffice.

## Example

```go
package main

import (
	"fmt"

	"github.com/go-erasure/mojette"
)

func main() {
	// 4x3 grid of 2-byte blocks.
	g := &mojette.Grid{Rows: 4, Cols: 3, BlockSize: 2}
	for c := 0; c < g.Rows*g.Cols; c++ {
		g.Data = append(g.Data, []byte{byte(c), byte(c * 3)})
	}

	dirs := []mojette.Direction{{P: -1, Q: 1}, {P: 0, Q: 1}, {P: 1, Q: 1}, {P: 2, Q: 1}}
	projs, err := mojette.Encode(g, dirs)
	if err != nil {
		panic(err)
	}

	// Lose one projection; the remaining three still satisfy Katz (sum|P| >= Cols).
	subset := []mojette.Projection{projs[0], projs[2], projs[3]}

	out, err := mojette.Reconstruct(g.Rows, g.Cols, g.BlockSize, subset)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(out.Data)) // 12 blocks recovered exactly
}
```

## API

```go
func Encode(g *Grid, dirs []Direction) ([]Projection, error)
func Reconstruct(rows, cols, blockSize int, projs []Projection) (*Grid, error)
func Reconstructible(rows, cols int, dirs []Direction) bool
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
