#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/grid/manifold_lib.h>

#include <fstream>
#include <iostream>

using namespace dealii;

template <int dim>
class Step5
{
public:
  Step5();
  void run();

private:
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};

template <int dim>
Step5<dim>::Step5()
  : fe(1)
  , dof_handler(triangulation)
{}

template <int dim>
void Step5<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


template <int dim>
void Step5<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0.;
      cell_rhs    = 0.;

      fe_values.reinit(cell);

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) +=
                  (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                   fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                   fe_values.JxW(q_index));           // dx

              cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                              0.1 *                               // f(x_q)
                              fe_values.JxW(q_index));            // dx
            }
        }


      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

  // With the matrix so built, we use zero boundary values again:
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           5,
                                           Functions::ConstantFunction<dim>(10.0),
                                           boundary_values);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           6,
                                           Functions::ConstantFunction<dim>(200.0),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
  
}

template <int dim>
void Step5<dim>::solve()
{
  SolverControl solver_control(1000, 1e-12);
  SolverCG<>    solver(solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}

template <int dim>
void Step5<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  data_out.build_patches();

  std::ofstream output("solution.vtk");
  data_out.write_vtk(output);
}

template <int dim>
void Step5<dim>::run()
{
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("engine2000.msh");
  grid_in.read_msh(input_file);
  
  Assert(dim == 2, ExcInternalError());

  // Taken from step-49 to print info about the mesh
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << triangulation.n_active_cells() << std::endl;
  {
    std::map<types::boundary_id, unsigned int> boundary_count;
    for (auto &cell : triangulation.active_cell_iterators())
      {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (cell->face(face)->at_boundary())
              boundary_count[cell->face(face)->boundary_id()]++;
          }
      }

    std::cout << " boundary indicators: ";
    for (const std::pair<const types::boundary_id, unsigned int> &pair :
         boundary_count)
      {
        std::cout << pair.first << "(" << pair.second << " times) ";
      }
    std::cout << std::endl;
  }

  std::ofstream out("grid-1.eps");
  GridOut       grid_out;
  grid_out.write_eps(triangulation, out);
  std::cout << " written to grid-1.eps\n\n";

  setup_system();
  assemble_system();
  solve();
  output_results();
}

int main()
{
  Step5<2> laplace_problem_2d;
  laplace_problem_2d.run();
  return 0;
}
