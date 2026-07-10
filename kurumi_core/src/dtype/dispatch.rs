//! Dtype-dispatch macros: a category op = (variant list, kernel body), expanded to one match
//! over the category's `Storage` variants. Adding a dtype = adding its variant to the lists
//! here, not editing N match arms. Brought crate-wide via `#[macro_use] mod dispatch` (in
//! `dtype.rs`) + `#[macro_use] mod dtype` (in `lib.rs`); the helper fns the bodies call
//! (`map1`/`zip_map`/`cmp_map`/`reduce_v`) resolve at each use site.

// dtype-dispatch helpers: a category op = (variant list, kernel body). Adding a dtype
// means adding its variant to the category lists below, not editing N match macros.
// `dispatch!` (all dtypes) stays an explicit exhaustive match so the compiler forces
// a new dtype to be handled there.
macro_rules! map1_variants {
    ($s:expr, [$($V:ident),+ $(,)?], $f:path) => {
        match $s { $(Storage::$V(v) => Storage::$V(map1(v, $f)),)+ _ => unreachable!("unary op outside dtype category") }
    };
}
macro_rules! pair_variants {
    ($a:expr, $b:expr, [$($V:ident),+ $(,)?], |$x:ident, $y:ident| $body:expr) => {
        match ($a, $b) { $((Storage::$V($x), Storage::$V($y)) => Storage::$V($body),)+ _ => unreachable!("binary op: dtype mismatch or outside category") }
    };
}
macro_rules! cmp_variants {
    ($a:expr, $b:expr, [$($V:ident),+ $(,)?], $f:path) => {
        match ($a, $b) { $((Storage::$V(x), Storage::$V(y)) => Storage::BOOL(cmp_map(x, y, $f)),)+ _ => unreachable!("cmp: dtype mismatch") }
    };
}
macro_rules! reduce_variants {
    ($s:expr, $shape:expr, $axis:expr, $init:path, $f:path, [$($V:ident),+ $(,)?]) => {
        match $s { $(Storage::$V(v) => reduce_v(v, $shape, $axis, $init(), $f),)+ _ => unreachable!("reduce on non-numeric dtype") }
    };
}

// map a storage to another of the SAME dtype; `$body` is a kernel generic over T.
// movement/elementwise reuse this; the kernel is written once, monomorphized per dtype.
macro_rules! dispatch {
    ($s:expr, |$v:ident| $body:expr) => {
        match $s {
            Storage::BOOL($v) => Storage::BOOL($body),
            Storage::U8($v) => Storage::U8($body),
            Storage::U16($v) => Storage::U16($body),
            Storage::U32($v) => Storage::U32($body),
            Storage::U64($v) => Storage::U64($body),
            Storage::I8($v) => Storage::I8($body),
            Storage::I16($v) => Storage::I16($body),
            Storage::I32($v) => Storage::I32($body),
            Storage::I64($v) => Storage::I64($body),
            Storage::F8E4M3($v) => Storage::F8E4M3($body),
            Storage::F8E5M2($v) => Storage::F8E5M2($body),
            Storage::F16($v) => Storage::F16($body),
            Storage::BF16($v) => Storage::BF16($body),
            Storage::F32($v) => Storage::F32($body),
            Storage::F64($v) => Storage::F64($body),
            Storage::C64($v) => Storage::C64($body),
            Storage::C128($v) => Storage::C128($body),
        }
    };
}

// category dispatches: each is a variant list + a body, fed to a helper above.
// numeric = all but BOOL; signed = ints+floats minus unsigned; int = integers;
// bitwise = bool+ints; cmp/where = all dtypes.
macro_rules! num_binary {
    ($a:expr, $b:expr, $f:path) => {
        pair_variants!(
            $a,
            $b,
            [U8, U16, U32, U64, I8, I16, I32, I64, F8E4M3, F8E5M2, F16, BF16, F32, F64, C64, C128],
            |x, y| zip_map(x, y, $f)
        )
    };
}
macro_rules! float_unary {
    ($s:expr, $f:path) => {
        map1_variants!($s, [F8E4M3, F8E5M2, F16, BF16, F32, F64, C64, C128], $f)
    };
}
macro_rules! signed_unary {
    ($s:expr, $f:path) => {
        map1_variants!($s, [I8, I16, I32, I64, F8E4M3, F8E5M2, F16, BF16, F32, F64, C64, C128], $f)
    };
}
macro_rules! num_reduce {
    ($s:expr, $shape:expr, $axis:expr, $init:path, $f:path) => {
        reduce_variants!(
            $s,
            $shape,
            $axis,
            $init,
            $f,
            [U8, U16, U32, U64, I8, I16, I32, I64, F8E4M3, F8E5M2, F16, BF16, F32, F64, C64, C128]
        )
    };
}
macro_rules! int_binary {
    ($a:expr, $b:expr, $f:path) => {
        pair_variants!($a, $b, [U8, U16, U32, U64, I8, I16, I32, I64], |x, y| zip_map(x, y, $f))
    };
}
macro_rules! bitwise_binary {
    ($a:expr, $b:expr, $f:path) => {
        pair_variants!($a, $b, [BOOL, U8, U16, U32, U64, I8, I16, I32, I64], |x, y| zip_map(x, y, $f))
    };
}
macro_rules! cmp_binary {
    ($a:expr, $b:expr, $f:path) => {
        cmp_variants!($a, $b, [BOOL, U8, U16, U32, U64, I8, I16, I32, I64, F8E4M3, F8E5M2, F16, BF16, F32, F64], $f)
    };
}
// same-dtype pair over ALL dtypes (for `where`); $body is generic over T
macro_rules! any_binary {
    ($a:expr, $b:expr, |$x:ident, $y:ident| $body:expr) => {
        pair_variants!(
            $a,
            $b,
            [BOOL, U8, U16, U32, U64, I8, I16, I32, I64, F8E4M3, F8E5M2, F16, BF16, F32, F64, C64, C128],
            |$x, $y| $body
        )
    };
}
