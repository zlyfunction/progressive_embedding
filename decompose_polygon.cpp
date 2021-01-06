#include "decompose_polygon.h"
#include "is_simple_polygon.h"
#include "embed_points.h"
#include "plot.h"
#include <igl/triangle_triangle_adjacency.h>
#include <igl/copyleft/cgal/orient2D.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/partition_2.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Partition_traits_2<K> Traits;
typedef Traits::Point_2 Point;
typedef Traits::Polygon_2 Polygon_2;

bool is_convex(
  const Eigen::MatrixXd& P
){
  for(int i=0;i<P.rows();i++){
    int prev = (i-1+P.rows())%P.rows();
    int next = (i+1)%P.rows();
    double a[2] = {P(prev,0),P(prev,1)};
    double b[2] = {P(i,0),P(i,1)};
    double c[2] = {P(next,0),P(next,1)};
    short r = igl::copyleft::cgal::orient2D(a,b,c);
    if(r < 0)
        return false;
  }
  return true;
}

void merge_triangles(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  std::vector<std::vector<int>>& L
){

  // use the greedy method for now
  // could be improved

  // collect halfedge and corresponding face id
  // initialize L to be collection of all faces(polygons)
  std::map<std::pair<int,int>, int> H;
  Eigen::VectorXi G(F.rows());
  Eigen::MatrixXi FF,FFI;
  igl::triangle_triangle_adjacency(F,FF,FFI);
  L.resize(F.rows());
  for(int i=0;i<F.rows();i++){
    G(i) = i; // every face belongs to itself
    for(int k=0;k<3;k++){
      L[i].push_back(F(i,k));
      H[std::make_pair(F(i,k),F(i,(k+1)%3))] = FF(i,k);
    }
  }

  // traverse all the halfedges
  for(auto h: H){ // [he, pid]
    auto he = h.first;
    auto rhe = std::make_pair(he.second,he.first);
    int p1 = H[he]; // a -> b
    int p2 = H[rhe]; // b -> a
    if(p1 == -1 || p2 == -1) continue;
    
    // up to group root
    while(p1!=G[p1])
      p1 = G[p1];
    while(p2!=G[p2])
      p2 = G[p2];

    // combine p1 and p2
    Eigen::MatrixXd poly(L[p1].size()+L[p2].size()-2,2);
    auto a = std::find(L[p1].begin(),L[p1].end(),he.first);
    auto b = std::find(L[p2].begin(),L[p2].end(),he.second);

    std::vector<int> L1(L[p1].begin(),a);
    std::vector<int> R1(a,L[p1].end());
    std::vector<int> L2(L[p2].begin(),b);
    std::vector<int> R2(b,L[p2].end());
    
    std::vector<int> S;
    S = L1;
    auto c = R2.empty() ? R2.begin() : R2.begin()+1;
    auto d = R1.empty() ? R1.begin() : R1.begin()+1;
    S.insert(S.end(),c,R2.end());
    S.insert(S.end(),L2.begin(),L2.end());
    S.insert(S.end(),d,R1.end());
      
    for(int i=0;i<poly.rows();i++){
      poly.row(i)<<V.row(S[i]);
    }
    
    // if merged polygon is simple, drop edge he/rhe
    // erase L[p2], add to L[p1]
    Polygon_2 pgn;
    for(int i = 0; i < poly.rows(); i++)
      pgn.push_back(Point(poly(i,0), poly(i,1)));
    
    // if(is_simple_polygon(poly) && is_convex(poly)){
    if(pgn.is_simple()){
      H[he]=-1;
      H[rhe]=-1;
      G[p2] = p1; // p2 belongs to p1 now
      L[p1] = S;
      L[p2].clear();
    }
  }

  // TODO: for(each simple poly defined in L){ // L[i] is a polygon, it contains the endpoints ids in V
    // - TODO: convert L to pgn, e.g. line 96
    // - try cgal partition on pgn
    std::vector<Polygon_2> partition_polys;
    CGAL::approx_convex_partition_2(pgn.vertices_begin(),
                                    pgn.vertices_end(),
                                    std::back_inserter(partition_polys));
    // visualize the polygons
    std::vector<Eigen::MatrixXd> P_sets; // for every edge in partition_polys we add two new points - just for vis
    for(int i = 0; i < partition_polys.size(); i++){
      auto poly = partition_polys[i];
      Eigen::MatrixXd P;
      int index = 0;
      for(auto it = poly.vertices_begin(); it != poly.vertices_end(); it++){
        P.conservativeResize(P.rows()+1, 2);
        P.row(index) << it->x(), it->y();
        index++;
      }
      P_sets.push_back(P);
      igl::opengl::glfw::Viewer viewer;
      Eigen::VectorXi T(P.rows());
      T.setConstant(1);
      viewer.data().set_mesh(V, F);
      plot_polygon(viewer, T, P);
      viewer.launch();
    }
    // TODO: convert the partition_polys back to L
  }

}

void decompose_polygon(
  const Eigen::MatrixXd& P,
  const Eigen::VectorXi& R,
  const Eigen::MatrixXd& C,
  Eigen::MatrixXd& V,
  Eigen::MatrixXi& F, 
  std::vector<std::vector<int>>& L
){
  bool succ = Shor_van_wyck(P,R,"",V,F,false);
  assert(succ && "Shor failed");
  embed_points(C,V,F);
  igl::opengl::glfw::Viewer vr;
  merge_triangles(V,F,L);
}
