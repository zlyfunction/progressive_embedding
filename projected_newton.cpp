#include "projected_newton.hpp"

#include <iostream>
#include <igl/Timer.h>

namespace jakob
{

#include "autodiff_jakob.h"
  DECLARE_DIFFSCALAR_BASE();

  double gradient_and_hessian_from_J(const Eigen::RowVector4d &J,
                                     Eigen::RowVector4d &local_grad,
                                     Eigen::Matrix4d &local_hessian)
  {
#ifdef NOHESSIAN
    using DScalar = DScalar1<double, Eigen::Vector4d>;
#else
    using DScalar = DScalar2<double, Eigen::Vector4d, Eigen::Matrix4d>;
#endif
    DiffScalarBase::setVariableCount(4);
    DScalar a(0, J(0));
    DScalar b(1, J(1));
    DScalar c(2, J(2));
    DScalar d(3, J(3));
    auto sd = symmetric_dirichlet_energy_t(a, b, c, d);

    local_grad = sd.getGradient();
#ifndef NOHESSIAN
    local_hessian = sd.getHessian();
#endif
    DiffScalarBase::setVariableCount(0);
    return sd.getValue();
  }
} // namespace jakob

namespace desai
{
#include "desai_symmd.c"
  double gradient_and_hessian_from_J(const Eigen::RowVector4d &J,
                                     Eigen::RowVector4d &local_grad,
                                     Eigen::Matrix4d &local_hessian)
  {
    double energy = symmetric_dirichlet_energy_t(J(0), J(1), J(2), J(3));
    double grad[4], hessian[10];
    reverse_diff(J.data(), 1, local_grad.data());
#ifndef NOHESSIAN
    reverse_hessian(J.data(), 1, local_hessian.data());
#endif
    return energy;
  }

  Eigen::VectorXd gradient_and_hessian_from_J_vec(const Eigen::Matrix<double, -1, 4, Eigen::RowMajor> &J,
                                                  Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &grad,
                                                  Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &hessian)
  {
    reverse_diff(J.data(), J.rows(), grad.data());
#ifndef NOHESSIAN
    reverse_hessian(J.data(), J.rows(), hessian.data());
    return symmetric_dirichlet_energy(J.col(0), J.col(1), J.col(2), J.col(3));
#endif
    return Eigen::VectorXd();
  }
} // namespace desai

double compute_energy_from_jacobian(const Xd &J, const Vd &area)
{
  return symmetric_dirichlet_energy(J.col(0), J.col(1), J.col(2), J.col(3)).dot(area) / area.sum();
  // return symmetric_dirichlet_energy(J.col(0), J.col(1), J.col(2), J.col(3)).dot(Eigen::VectorXd::Ones(area.rows())) / area.rows(); // uniform
}

extern long global_autodiff_time;
extern long global_project_time;
double grad_and_hessian_from_jacobian(const Vd &area, const Xd &jacobian,
                                      Xd &total_grad, spXd &hessian)
{
  int f_num = area.rows();
  total_grad.resize(f_num, 4);
  total_grad.setZero();
  double energy = 0;
  hessian.resize(4 * f_num, 4 * f_num);
  std::vector<Eigen::Triplet<double>> IJV;
  IJV.reserve(16 * f_num);
  double total_area = area.sum();

  std::vector<Eigen::Matrix4d> all_hessian(f_num);
  igl::Timer timer;
  timer.start();
  // #ifndef AD_ENGINE
  //   Eigen::Matrix<double, -1, -1, Eigen::RowMajor> half_hessian(f_num, 16);
  //   Eigen::Matrix<double, -1, -1, Eigen::RowMajor> local_grad(f_num, 4);
  //   Vd energy_vec = desai::gradient_and_hessian_from_J_vec(jacobian, local_grad, half_hessian);
  // #ifndef NOHESSIAN
  //   energy = energy_vec.dot(area) / total_area;
  //   total_grad = area.asDiagonal() * local_grad / total_area;
  //   half_hessian = area.asDiagonal() * half_hessian / total_area;
  //   for (int i = 0; i < f_num; i++)
  //   {
  //     auto hessian = half_hessian.row(i);
  //     all_hessian[i] << hessian[0], hessian[1], hessian[2], hessian[3],
  //         hessian[1], hessian[4], hessian[5], hessian[6],
  //         hessian[2], hessian[5], hessian[7], hessian[8],
  //         hessian[3], hessian[6], hessian[8], hessian[9];
  //   }
  // #endif
  // #else
  for (int i = 0; i < f_num; i++)
  {
    Eigen::RowVector4d J = jacobian.row(i);
    Eigen::Matrix4d local_hessian;
    Eigen::RowVector4d local_grad;
    energy += AD_ENGINE::gradient_and_hessian_from_J(J, local_grad, local_hessian) * area(i) / total_area;
    // energy += AD_ENGINE::gradient_and_hessian_from_J(J, local_grad, local_hessian) / f_num;

#ifndef NOHESSIAN
    local_grad *= area(i) / total_area;
    local_hessian *= area(i) / total_area;
    all_hessian[i] = local_hessian;
    total_grad.row(i) = local_grad;
#endif
  }
  // #endif
  global_autodiff_time += timer.getElapsedTimeInMicroSec();

#ifndef NOHESSIAN
  hessian.reserve(Eigen::VectorXi::Constant(4 * f_num, 4));
  for (int i = 0; i < f_num; i++)
  {
    Eigen::Matrix4d &local_hessian = all_hessian[i];
    if (fabs(total_grad(i)) > 1e-3) project_hessian(local_hessian);
    for (int v1 = 0; v1 < 4; v1++)
      for (int v2 = 0; v2 < v1 + 1; v2++)
        hessian.insert(v1 * f_num + i, v2 * f_num + i) = local_hessian(v1, v2);
  }
  hessian.makeCompressed();
  // spXd id(4 * f_num, 4 * f_num);
  // id.setIdentity();
  // hessian = hessian + *id;//Eigen::DiagonalMatrix<double>::Identity();
#endif
  return energy;
}

void jacobian_from_uv(const spXd &G, const Xd &uv, Xd &Ji)
{
  Vd altJ = G * Eigen::Map<const Vd>(uv.data(), uv.size());
  Ji = (Xd)Eigen::Map<Xd>(altJ.data(), G.rows() / 4, 4);
}

Vd vec(Xd &M2)
{
  Vd v = Eigen::Map<Vd>(M2.data(), M2.size());
  return v;
}

double get_grad_and_hessian(const spXd &G, const Vd &area, const Xd &uv,
                            Vd &grad, spXd &hessian)
{
  int f_num = area.rows();
  Xd Ji, total_grad;
  jacobian_from_uv(G, uv, Ji);
  double energy;
  energy = grad_and_hessian_from_jacobian(area, Ji, total_grad, hessian);

  Vd vec_grad = vec(total_grad); //+2 * lambda * x_i
  hessian = G.transpose() * hessian.selfadjointView<Eigen::Lower>() * G;  // +2 * lambda*Id
  grad = vec_grad.transpose() * G;

  return energy;
}


#include <igl/copyleft/cgal/orient2D.h>
int check_flip(const Eigen::MatrixXd &uv, const Eigen::MatrixXi &Fn)
{
  int fl = 0;
  // std::cout << "uv" << uv.rows() << std::endl;
  // std::cout << uv << std::endl;
  for (int i = 0; i < Fn.rows(); i++)
  {
    // std::cout << "Fn.row(i) = "<< Fn.row(i) << std::endl;
    double a[2] = {uv(Fn(i, 0), 0), uv(Fn(i, 0), 1)};
    double b[2] = {uv(Fn(i, 1), 0), uv(Fn(i, 1), 1)};
    double c[2] = {uv(Fn(i, 2), 0), uv(Fn(i, 2), 1)};
    // std::cout << a[0] << " " << a[1] << std::endl;
    // std::cout << b[0] << " " << b[1] << std::endl;
    // std::cout << c[0] << " " << c[1] << std::endl;
    if (igl::copyleft::cgal::orient2D(a, b, c) <= 0)
    {
      // std::cout << "flip @ triangle: " << i << std::endl;
      fl++;
    }
  }
  return fl;
  // std::cout << "flipped # " << fl << std::endl;
}
#include <igl/flip_avoiding_line_search.h>
#include <iostream>
double bi_linesearch(
    const Eigen::MatrixXi F,
    Eigen::MatrixXd &cur_v,
    Eigen::MatrixXd &d,
    std::function<double(Eigen::MatrixXd &)> energy,
    // std::function<Eigen::VectorXd(Eigen::MatrixXd &)> get_grad,
    Eigen::VectorXd &grad0,
    double energy0)
{
  double step_size = 2.0;
  // step_size = 1.01;
  double new_energy = 0;
  Eigen::MatrixXd newx;
  Vd flat_d = Eigen::Map<const Vd>(d.data(), d.size());
  double slope = flat_d.dot(grad0);
  double c1 = 1e-4;
  while (true)
  {
    step_size /= 2;
    // step_size -= 0.01;
    newx = cur_v + step_size * d;
    if (check_flip(newx, F) > 0)
    {
      // std::cout << "cause flip, step_size/=2\n";
      continue;
    }
    new_energy = energy(newx);

    // test line search
    // std::cout << "step_size = " << step_size << "\t";
    // std::cout << "energy0 = " << energy0 <<"\tnew_energy = " << new_energy << "\n";
    // Eigen::VectorXd new_gradE = get_grad(newx);
    // std::cout << "grad.dot(d) = " << new_gradE.dot(flat_d) << "\t";
    // std::cout << "grad:\n" << new_gradE << std::endl;
    // Xd newx_shift = newx + 1e-6 * d;
    // std::cout << "de/ds = " << (energy(newx_shift) - new_energy) / 1e-6 << std::endl;

    // if (new_energy <= energy0 + c1 * step_size * slope) // armijo
    // {
    //   break;
    // }
    if ((new_energy < energy0))
    {
      break;
    }
    if (step_size == 0)
    {
      break;
      // return new_energy;
    }
    // std::cout << "energy did not decrease, step_size/=2\n";
  }
  // std::cout << "step size: " << step_size << std::endl;
  cur_v = newx;
  return new_energy;
}

double wolfe_linesearch(
    const Eigen::MatrixXi F,
    Eigen::MatrixXd &cur_v,
    Eigen::MatrixXd &d,
    std::function<double(Eigen::MatrixXd &)> energy,
    Eigen::VectorXd &grad0,
    double energy0, bool use_gd)
{
  using namespace std;

  double min_step_to_singularity = igl::flip_avoiding::compute_max_step_from_singularities(cur_v, F, d);
  double step_size = std::min(1., min_step_to_singularity * 0.8);
  // std::cout << "min_step_to_singularity:" << min << std::endl;
  double new_energy = 0;
  Eigen::MatrixXd newx;
  Vd flat_d = Eigen::Map<const Vd>(d.data(), d.size());
  double slope = flat_d.dot(grad0);
  auto c1 = 1e-4;
  for (int i = 0; i < 200; i++)
  {
    newx = cur_v + step_size * d;
    new_energy = energy(newx);
    if (new_energy <= energy0 + c1 * step_size * slope // armijo
    )
      break;
    if ((use_gd) && (new_energy < energy0))
    {
      break;
    }

    step_size = 0.8 * step_size;
  }
  std::cout << "step size: " << step_size << std::endl;
  // check stepsize
  cur_v = newx;
  return new_energy;
}
