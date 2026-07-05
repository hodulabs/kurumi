use kurumi_core::*;

#[test]
fn fft_basics() {
    let mut g = Graph::new();
    // delta -> all ones
    let d = g.constant(vec![1., 0., 0., 0.], vec![4]);
    let fd = g.fft(d, 0).unwrap();
    let (fr, fi) = (g.real(fd).unwrap(), g.imag(fd).unwrap());
    assert_eq!(interpret(&g, fr).f32(), &[1., 1., 1., 1.]);
    assert_eq!(interpret(&g, fi).f32(), &[0., 0., 0., 0.]);
    // constant signal -> [n, 0, 0, 0]
    let o = g.constant(vec![1., 1., 1., 1.], vec![4]);
    let fo = g.fft(o, 0).unwrap();
    let for_ = g.real(fo).unwrap();
    let fov = interpret(&g, for_).f32().to_vec();
    assert!((fov[0] - 4.).abs() < 1e-4 && fov[1..].iter().all(|&x| x.abs() < 1e-4), "fft ones {fov:?}");
    // round-trip: ifft(fft(x)) = x
    let x = g.constant(vec![1., 2., 3., 4.], vec![4]);
    let rt = {
        let f = g.fft(x, 0).unwrap();
        g.ifft(f, 0).unwrap()
    };
    let rtr = g.real(rt).unwrap();
    let rtv = interpret(&g, rtr).f32().to_vec();
    for (got, want) in rtv.iter().zip([1., 2., 3., 4.]) {
        assert!((got - want).abs() < 1e-4, "roundtrip {rtv:?}");
    }
}
