package mojette

import (
	"bytes"
	"errors"
	"testing"
)

// makeGrid builds a rows*cols grid of blockSize-byte blocks with deterministic,
// distinct content so an exact round-trip comparison is meaningful.
func makeGrid(rows, cols, blockSize int) *Grid {
	data := make([][]byte, rows*cols)
	for c := range data {
		blk := make([]byte, blockSize)
		for b := range blk {
			blk[b] = byte(c*7 + b*13 + 1)
		}
		data[c] = blk
	}
	return &Grid{Rows: rows, Cols: cols, BlockSize: blockSize, Data: data}
}

func gridsEqual(a, b *Grid) bool {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.BlockSize != b.BlockSize {
		return false
	}
	if len(a.Data) != len(b.Data) {
		return false
	}
	for i := range a.Data {
		if !bytes.Equal(a.Data[i], b.Data[i]) {
			return false
		}
	}
	return true
}

func TestReconstructibleKatz(t *testing.T) {
	// sumQ >= rows.
	if !Reconstructible(4, 3, []Direction{{0, 1}, {1, 1}, {1, 1}, {1, 1}}) {
		t.Fatal("expected reconstructible via sumQ")
	}
	// sumP >= cols (sumQ falls short).
	if !Reconstructible(4, 3, []Direction{{-1, 1}, {1, 1}, {2, 1}}) {
		t.Fatal("expected reconstructible via sumP")
	}
	// Neither: single vertical projection.
	if Reconstructible(4, 3, []Direction{{0, 1}}) {
		t.Fatal("expected not reconstructible")
	}
}

func TestEncodeReconstructRoundTripDropOne(t *testing.T) {
	// Spec example: 4x3 grid, directions with both P<0 and P>0 (covers both
	// minRaw branches). Drop one projection (0,1); the remainder satisfies the
	// Katz criterion via sum(|P|) = 4 >= Cols and reconstructs exactly.
	dirs := []Direction{{-1, 1}, {0, 1}, {1, 1}, {2, 1}}
	g := makeGrid(4, 3, 2)
	projs, err := Encode(g, dirs)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(projs) != len(dirs) {
		t.Fatalf("got %d projections, want %d", len(projs), len(dirs))
	}

	subDirs := []Direction{{-1, 1}, {1, 1}, {2, 1}}
	if !Reconstructible(4, 3, subDirs) {
		t.Fatal("subset should satisfy Katz")
	}
	subset := []Projection{projs[0], projs[2], projs[3]}
	got, err := Reconstruct(4, 3, 2, subset)
	if err != nil {
		t.Fatalf("reconstruct (drop one): %v", err)
	}
	if !gridsEqual(got, g) {
		t.Fatal("reconstruct (drop one) mismatch")
	}
}

func TestEncodeReconstructRoundTripDropTwo(t *testing.T) {
	// 2x2 grid, BlockSize>1. Encode four directions, then reconstruct from a
	// two-direction subset {(0,1),(1,1)}: sum(Q) = 2 >= Rows satisfies Katz and
	// the pair reconstructs exactly (dropping (-1,1) and (2,1)).
	dirs := []Direction{{-1, 1}, {0, 1}, {1, 1}, {2, 1}}
	g := makeGrid(2, 2, 3)
	projs, err := Encode(g, dirs)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	subDirs := []Direction{{0, 1}, {1, 1}}
	if !Reconstructible(2, 2, subDirs) {
		t.Fatal("subset should satisfy Katz")
	}
	subset := []Projection{projs[1], projs[2]}
	got, err := Reconstruct(2, 2, 3, subset)
	if err != nil {
		t.Fatalf("reconstruct (drop two): %v", err)
	}
	if !gridsEqual(got, g) {
		t.Fatal("reconstruct (drop two) mismatch")
	}
}

func TestReconstructNotReconstructible(t *testing.T) {
	g := makeGrid(4, 3, 1)
	projs, err := Encode(g, []Direction{{0, 1}})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if _, err := Reconstruct(4, 3, 1, projs); !errors.Is(err, ErrNotReconstructible) {
		t.Fatalf("got %v, want ErrNotReconstructible", err)
	}
}

func TestEncodeInvalidGrid(t *testing.T) {
	dirs := []Direction{{0, 1}}
	cases := []struct {
		name string
		g    *Grid
	}{
		{"zero rows", &Grid{Rows: 0, Cols: 3, BlockSize: 1, Data: nil}},
		{"zero cols", &Grid{Rows: 3, Cols: 0, BlockSize: 1, Data: nil}},
		{"zero block", &Grid{Rows: 3, Cols: 3, BlockSize: 0, Data: nil}},
		{"wrong data len", &Grid{Rows: 2, Cols: 2, BlockSize: 1, Data: [][]byte{{1}}}},
		{"wrong block len", &Grid{Rows: 1, Cols: 2, BlockSize: 2, Data: [][]byte{{1, 2}, {3}}}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := Encode(c.g, dirs); !errors.Is(err, ErrInvalidGrid) {
				t.Fatalf("got %v, want ErrInvalidGrid", err)
			}
		})
	}
}

func TestEncodeInvalidDirection(t *testing.T) {
	g := makeGrid(2, 2, 1)
	cases := []struct {
		name string
		d    Direction
	}{
		{"q below one", Direction{P: 1, Q: 0}},
		{"gcd not one", Direction{P: 2, Q: 4}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := Encode(g, []Direction{c.d}); !errors.Is(err, ErrInvalidDirection) {
				t.Fatalf("got %v, want ErrInvalidDirection", err)
			}
		})
	}
}

func TestReconstructInvalidDims(t *testing.T) {
	if _, err := Reconstruct(0, 3, 1, nil); !errors.Is(err, ErrInvalidGrid) {
		t.Fatalf("got %v, want ErrInvalidGrid", err)
	}
}

func TestReconstructInvalidDirection(t *testing.T) {
	p := Projection{Dir: Direction{P: 2, Q: 4}, Bins: nil}
	if _, err := Reconstruct(2, 2, 1, []Projection{p}); !errors.Is(err, ErrInvalidDirection) {
		t.Fatalf("got %v, want ErrInvalidDirection", err)
	}
}

func TestReconstructMismatchedBins(t *testing.T) {
	// Correct nbins for (0,1) on 2x2 is 2; supply the wrong count.
	short := Projection{Dir: Direction{P: 0, Q: 1}, Bins: [][]byte{{0}}}
	if _, err := Reconstruct(2, 2, 1, []Projection{short}); !errors.Is(err, ErrInvalidGrid) {
		t.Fatalf("short bins: got %v, want ErrInvalidGrid", err)
	}

	// Right bin count, but a block of the wrong size.
	badBlock := Projection{Dir: Direction{P: 0, Q: 1}, Bins: [][]byte{{0, 0}, {0}}}
	if _, err := Reconstruct(2, 2, 1, []Projection{badBlock}); !errors.Is(err, ErrInvalidGrid) {
		t.Fatalf("bad block: got %v, want ErrInvalidGrid", err)
	}
}
