import sympy as sp


def filter_poly_terms(poly, tol):
    if not isinstance(poly, sp.Poly):
        raise ValueError("Input must be a SymPy Poly.")
    new_poly = sp.Poly(0, *poly.gens)
    for monom, coeff in poly.as_dict().items():
        if abs(coeff) >= tol:
            new_poly += sp.Mul(*[gen**deg for gen, deg in zip(poly.gens, monom)], coeff)
    return new_poly


def remove_high_degree_terms(poly, vars, max_degree):
    # Check if the input is a SymPy Poly
    if not isinstance(poly, sp.Poly):
        raise ValueError("Input must be a SymPy Poly.")

    # Normalize vars to a set of SymPy symbols
    if isinstance(vars, sp.Symbol):
        vars = {vars}
    vars = set(vars)  # Ensure vars is a set of variables

    # Rebuild the polynomial without high total degree terms in specified variables
    new_poly = sp.Poly(
        0, *poly.gens
    )  # Start with a zero polynomial with the same generators
    for monom, coeff in poly.as_dict().items():
        # Calculate the total degree for the specified variables
        total_degree = sum(
            degree for gen, degree in zip(poly.gens, monom) if gen in vars
        )
        if total_degree <= max_degree:
            # Only add the term if its total degree in the specified variables is <= max_degree
            new_poly += sp.Mul(*[gen**deg for gen, deg in zip(poly.gens, monom)], coeff)

    return new_poly


def poisson_bracket(H1, H2, dim):
    p = sp.symbols(f"x:{dim}")
    q = sp.symbols(f"x{dim}:{2 * dim}")
    return sum(
        [
            sp.diff(H1, p[i]) * sp.diff(H2, q[i])
            - sp.diff(H1, q[i]) * sp.diff(H2, p[i])
            for i in range(dim)
        ]
    )


def inv_bea(H1, H2, dim, h, order, tol=None, truncate=False, max_degree=None):
    """two layer sympnet backward error map to the inverse modified hamiltonian
    from an order one splitting method"""
    assert order >= 0

    correction = H1 + H2
    if order == 0:
        return correction

    def PB(H1, H2):
        return poisson_bracket(H1, H2, dim)

    # calculate first correction
    if order >= 1:
        PB12 = PB(H1, H2)
        correction += h / 2 * PB12

    # calculate higher order corrections
    if order >= 2:
        correction += h**2 / 12 * (PB(H1 - H2, PB12))

    if order >= 3:
        correction -= (
            h**3 / 24 * PB(H2, PB(H1, PB12))
        )
    
    if order >= 4:
        PB22221 = PB(H2, PB(H2, PB(H2, PB(H2, H1))))
        PB11112 = PB(H1, PB(H1, PB(H1, PB(H1, H2))))
        PB12221 = PB(H1, PB(H2, PB(H2, PB(H2, H1))))
        PB21112 = PB(H2, PB(H1, PB(H1, PB(H1, H2))))
        PB21212 = PB(H2, PB(H1, PB(H2, PB(H1, H2))))
        PB12121 = PB(H1, PB(H2, PB(H1, PB(H2, H1))))
        correction -= h**4 / 720 * (PB22221 + PB11112)
        correction += h**4 / 360 * (PB12221 + PB21112)
        correction += h**4 / 120 * (PB21212 + PB12121)
    
    if order >= 5:
        PB121212 = PB(H1, PB(H2, PB(H1, PB(H2, PB(H1, H2)))))
        correction += h**5 / 240 * (PB121212)

    if tol is not None:
        correction = filter_poly_terms(correction, tol=tol)

    vars = {var for var in H1.gens if var != h}
    if truncate:
        assert (
            max_degree is not None
        ), "must provide max degree of the basis polynomials if truncate=true"
        correction = remove_high_degree_terms(correction, vars=h, max_degree=order)
        correction = remove_high_degree_terms(
            correction, vars=vars, max_degree=max_degree + 1
        )

    return correction


def truncated_bea_map(
    sub_hamiltonian_list_sym, dim, h, order, x_vars, truncate=False, tol=None
):
    """corrections to the inverse modified hamiltonian from an order one
    splitting method with n splittings"""
    if order == 0:
        return sum(sub_hamiltonian_list_sym)

    max_degree = sp.Poly(sub_hamiltonian_list_sym[0]).total_degree()

    vars = x_vars + (h,)
    H_temp = sp.Poly(0, *vars)
    for H in sub_hamiltonian_list_sym:
        H_temp = inv_bea(
            sp.Poly(H_temp, *vars),
            sp.Poly(H, *vars),
            dim,
            h,
            order=order,
            truncate=truncate,
            tol=tol,
            max_degree=max_degree,
        )
    return H_temp

    # x, p, q = sp.symbols(f"x:{2*dim}"), sp.symbols(f"p:{dim}"), sp.symbols(f"q:{dim}")
    # true_hamiltonian, dim = get_hamiltonian(config["ODE_NAME"])
    # true_hamiltonian = true_hamiltonian.subs({x[i]: p[i] for i in range(dim)})
    # true_hamiltonian = true_hamiltonian.subs({x[i + dim]: q[i] for i in range(dim)})

