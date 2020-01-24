#section support_code_apply

int APPLY_SPECIFIC(matmul)(PyArrayObject* input0, PyArrayObject* input1, PyArrayObject* input2,
                           PyArrayObject* input3, PyArrayObject* input4, PyArrayObject** output0, 
                           PyArrayObject** output1, PyArrayObject** output2) {
    
  using namespace exoplanet;

  int success = 0;
  npy_intp N, J, N2, Nrhs;
  auto U_in = get_matrix_input<DTYPE_INPUT_1>(&N, &J, input1, &success);
  if (success) return 1;
  if (CELERITE_J != Eigen::Dynamic && J != CELERITE_J) {
    PyErr_Format(PyExc_ValueError, "runtime value of J does not match compiled value");
    return 1;
  }

  auto Z_in = get_matrix_input<DTYPE_INPUT_4>(&N2, &Nrhs, input4, &success);
  if (success) return 1;
  if (N != N2) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }
  if (CELERITE_NRHS != Eigen::Dynamic && Nrhs != CELERITE_NRHS) {
    PyErr_Format(PyExc_ValueError, "runtime value of n_rhs does not match compiled value");
    return 1;
  }

  npy_intp input0_shape[] = {N};
  npy_intp input2_shape[] = {N, J};
  npy_intp input3_shape[] = {N-1, J};
  auto a_in = get_input<DTYPE_INPUT_0>(1, input0_shape, input0, &success);
  auto V_in = get_input<DTYPE_INPUT_2>(2, input2_shape, input2, &success);
  auto P_in = get_input<DTYPE_INPUT_3>(2, input3_shape, input3, &success);
  if (success) return 1;

  npy_intp shape0[] = {N, Nrhs};
  npy_intp shape1[] = {J, Nrhs};
  npy_intp shape2[] = {J, Nrhs};
  auto Y_out = allocate_output<DTYPE_OUTPUT_0>(2, shape0, TYPENUM_OUTPUT_0, output0, &success);
  auto F_plus_out = allocate_output<DTYPE_OUTPUT_1>(2, shape1, TYPENUM_OUTPUT_1, output1, &success);
  auto F_minus_out = allocate_output<DTYPE_OUTPUT_2>(2, shape2, TYPENUM_OUTPUT_2, output2, &success);
  if (success) return 1;

  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, 1>> a(a_in, N);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, 
    CELERITE_J, CELERITE_J_ORDER>> U(U_in, N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_2, Eigen::Dynamic, 
    CELERITE_J, CELERITE_J_ORDER>> V(V_in, N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_3, Eigen::Dynamic, 
    CELERITE_J, CELERITE_J_ORDER>> P(P_in, N-1, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_4, Eigen::Dynamic, 
    CELERITE_NRHS, CELERITE_NRHS_ORDER>> Z(Z_in, N, Nrhs);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, 
    CELERITE_NRHS, CELERITE_NRHS_ORDER>> Y(Y_out, N, Nrhs);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_1, Eigen::Dynamic, 
    CELERITE_NRHS, CELERITE_NRHS_ORDER>> F_plus(F_plus_out, J, Nrhs);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_2, Eigen::Dynamic, 
    CELERITE_NRHS, CELERITE_NRHS_ORDER>> F_minus(F_minus_out, J, Nrhs);
  
  Y.setZero();
  F_plus.setZero();
  F_minus.setZero();
  celerite::matmul(a, U, V, P, Z, Y, F_plus, F_minus);

  return 0;
}