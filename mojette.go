// Package mojette implements the Mojette transform erasure code: the discrete
// Radon-projection code used by RozoFS, operating XOR-wise (GF(2) addition)
// over fixed-size byte blocks.
//
// Data is arranged as a Grid of Rows*Cols blocks in row-major order, each block
// being BlockSize bytes. Each Direction (P, Q) yields one Projection built by
// summing (XOR-ing) grid blocks that fall onto the same discrete line. Given a
// Katz-sufficient subset of projections, the grid can be rebuilt with the
// iterative inverse Mojette (back-projection) algorithm.
package mojette

import "errors"

// Direction is a Mojette projection direction (P, Q). It requires Q >= 1 and
// gcd(|P|, Q) == 1.
type Direction struct{ P, Q int }

// Grid holds Rows*Cols data blocks in row-major order, each block BlockSize
// bytes long.
type Grid struct {
	Rows, Cols int
	BlockSize  int
	Data       [][]byte // len == Rows*Cols, each block == BlockSize bytes
}

// Projection is the Mojette projection of a grid along Dir. Bins has one block
// per discrete line; its length is the number of bins of Dir on the grid.
type Projection struct {
	Dir  Direction
	Bins [][]byte // len == nbins for Dir on the grid; each block BlockSize bytes
}

var (
	// ErrInvalidGrid reports a grid (or reconstruction target) whose
	// dimensions or block sizes are inconsistent.
	ErrInvalidGrid = errors.New("mojette: invalid grid dimensions or block sizes")
	// ErrInvalidDirection reports a direction violating Q>=1 and gcd(|P|,Q)=1.
	ErrInvalidDirection = errors.New("mojette: direction requires Q>=1 and gcd(|P|,Q)=1")
	// ErrNotReconstructible reports that the supplied projections are not
	// sufficient to rebuild the grid.
	ErrNotReconstructible = errors.New("mojette: projections insufficient to reconstruct")
)

// abs returns the absolute value of x.
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// gcd returns the greatest common divisor of a and b (a, b >= 0).
func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// validDirection reports whether d satisfies Q>=1 and gcd(|P|,Q)==1.
func validDirection(d Direction) bool {
	return d.Q >= 1 && gcd(abs(d.P), d.Q) == 1
}

// validDims reports whether the given dimensions are all positive.
func validDims(rows, cols, blockSize int) bool {
	return rows > 0 && cols > 0 && blockSize > 0
}

// nbins returns the number of bins of direction d on a rows*cols grid:
//
//	nbins = Q*(rows-1) + |P|*(cols-1) + 1
func nbins(rows, cols int, d Direction) int {
	return d.Q*(rows-1) + abs(d.P)*(cols-1) + 1
}

// binIndex returns the bin index of grid cell (i, j) for direction d:
//
//	raw    = Q*i - P*j
//	minRaw = (P > 0) ? -P*(cols-1) : 0
//	bin    = raw - minRaw
func binIndex(cols int, d Direction, i, j int) int {
	raw := d.Q*i - d.P*j
	minRaw := 0
	if d.P > 0 {
		minRaw = -d.P * (cols - 1)
	}
	return raw - minRaw
}

// xorBlock XORs src into dst block-wise (GF(2) addition). It is isolated so a
// SIMD variant can replace it later; dst and src must have equal length.
func xorBlock(dst, src []byte) {
	for i := range dst {
		dst[i] ^= src[i]
	}
}

// Reconstructible reports whether dirs satisfy the Katz criterion for a
// rows*cols grid:
//
//	sum(Q_k) >= rows  OR  sum(|P_k|) >= cols
func Reconstructible(rows, cols int, dirs []Direction) bool {
	sumQ, sumP := 0, 0
	for _, d := range dirs {
		sumQ += d.Q
		sumP += abs(d.P)
	}
	return sumQ >= rows || sumP >= cols
}

// Encode computes one projection per direction. It returns ErrInvalidGrid for a
// malformed grid and ErrInvalidDirection for a direction that violates Q>=1 or
// gcd(|P|,Q)==1.
func Encode(g *Grid, dirs []Direction) ([]Projection, error) {
	if !validDims(g.Rows, g.Cols, g.BlockSize) {
		return nil, ErrInvalidGrid
	}
	if len(g.Data) != g.Rows*g.Cols {
		return nil, ErrInvalidGrid
	}
	for _, blk := range g.Data {
		if len(blk) != g.BlockSize {
			return nil, ErrInvalidGrid
		}
	}
	for _, d := range dirs {
		if !validDirection(d) {
			return nil, ErrInvalidDirection
		}
	}

	projs := make([]Projection, len(dirs))
	for k, d := range dirs {
		nb := nbins(g.Rows, g.Cols, d)
		bins := make([][]byte, nb)
		for b := range bins {
			bins[b] = make([]byte, g.BlockSize)
		}
		for i := 0; i < g.Rows; i++ {
			for j := 0; j < g.Cols; j++ {
				xorBlock(bins[binIndex(g.Cols, d, i, j)], g.Data[i*g.Cols+j])
			}
		}
		projs[k] = Projection{Dir: d, Bins: bins}
	}
	return projs, nil
}

// Reconstruct rebuilds the rows*cols grid (blocks of blockSize) from a subset of
// projections using the iterative inverse Mojette (back-projection) over XOR. It
// returns ErrInvalidGrid for bad dimensions or mismatched projection bins,
// ErrInvalidDirection for a bad projection direction, and ErrNotReconstructible
// when the projections do not suffice.
func Reconstruct(rows, cols, blockSize int, projs []Projection) (*Grid, error) {
	if !validDims(rows, cols, blockSize) {
		return nil, ErrInvalidGrid
	}
	for _, p := range projs {
		if !validDirection(p.Dir) {
			return nil, ErrInvalidDirection
		}
		if len(p.Bins) != nbins(rows, cols, p.Dir) {
			return nil, ErrInvalidGrid
		}
		for _, blk := range p.Bins {
			if len(blk) != blockSize {
				return nil, ErrInvalidGrid
			}
		}
	}

	total := rows * cols
	data := make([][]byte, total)
	solved := make([]bool, total)
	solvedCount := 0

	// Mutable copies of every provided projection's bins, plus per-projection
	// cell->bin maps and per-bin counts of still-unknown contributing cells.
	bins := make([][][]byte, len(projs))
	cellBin := make([][]int, len(projs))
	unknown := make([][]int, len(projs))
	for k, p := range projs {
		bins[k] = make([][]byte, len(p.Bins))
		for b := range p.Bins {
			blk := make([]byte, blockSize)
			copy(blk, p.Bins[b])
			bins[k][b] = blk
		}
		cellBin[k] = make([]int, total)
		unknown[k] = make([]int, len(p.Bins))
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				b := binIndex(cols, p.Dir, i, j)
				cellBin[k][i*cols+j] = b
				unknown[k][b]++
			}
		}
	}

	for solvedCount < total {
		progressed := false
		for k := range projs {
			for b := range unknown[k] {
				if unknown[k][b] != 1 {
					continue
				}
				// The single still-unknown contributor to bin b equals the
				// bin's current (fully back-projected) value.
				cell := 0
				for ; cell < total; cell++ {
					if !solved[cell] && cellBin[k][cell] == b {
						break
					}
				}
				val := make([]byte, blockSize)
				copy(val, bins[k][b])
				data[cell] = val
				solved[cell] = true
				solvedCount++
				for m := range projs {
					bb := cellBin[m][cell]
					xorBlock(bins[m][bb], val)
					unknown[m][bb]--
				}
				progressed = true
			}
		}
		if !progressed {
			return nil, ErrNotReconstructible
		}
	}

	return &Grid{Rows: rows, Cols: cols, BlockSize: blockSize, Data: data}, nil
}
