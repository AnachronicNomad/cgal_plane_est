#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <boost/tuple/tuple.hpp>
#include <CGAL/Point_3.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Vector_3.h>
#include <CGAL/Plane_3.h>

#include <CGAL/property_map.h>

#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/wlop_simplify_and_regularize_point_set.h>
#include <CGAL/property_map.h>

#include <CGAL/hilbert_sort.h>
#include <CGAL/spatial_sort.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>

#include <vector>
#include <fstream>

#include <CGAL/jet_smooth_point_set.h>
#include <CGAL/jet_estimate_normals.h>
//#include <CGAL/mst_orient_normals.h>

#include <CGAL/Shape_detection/Efficient_RANSAC.h>

#include <CGAL/property_map.h>
#include <CGAL/IO/write_ply_points.h>

#include <CGAL/Real_timer.h>

#include <CGAL/Shape_detection/Efficient_RANSAC.h>

#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing_on_point_set.h>

#include <CGAL/squared_distance_3.h>

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>

#include <boost/graph/adjacency_list.hpp>
#include <CGAL/boost/graph/kruskal_min_spanning_tree.h>
#include <map>

#include <typeinfo>

// types
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
//using FT = Kernel::FT;

using Point_3 = Kernel::Point_3;
using Point_Set = CGAL::Point_set_3<Point_3>;
using Vector_3 = Kernel::Vector_3;
using Plane_3 = Kernel::Plane_3;

using RGBA = std::array<uint16_t, 3>;
using PointData = boost::tuple<
    Point_3,    // Point location; Easting-Altitude-Northing
    Vector_3,   // Estimated Normal Vector at Point
    RGBA,       // Photo Color
    RGBA,       // RANSAC Shape Colorization
    size_t,     // Estimated Patch ID
    RGBA,       // Estimated Patch Colorization
    size_t>;    // orderering in the container? 

using PointMap = CGAL::Nth_of_tuple_property_map<0, PointData>;
using NormalMap = CGAL::Nth_of_tuple_property_map<1, PointData>;
using PhotoColorMap = CGAL::Nth_of_tuple_property_map<2, PointData>;
using ShapeColorMap = CGAL::Nth_of_tuple_property_map<3, PointData>;
using PatchIDMap = CGAL::Nth_of_tuple_property_map<4, PointData>;
using PatchColorMap = CGAL::Nth_of_tuple_property_map<5, PointData>;

using RANSACTraits = CGAL::Shape_detection::Efficient_RANSAC_traits<
    Kernel, 
    std::vector<PointData>,
    PointMap,
    NormalMap
>;
using EfficientRANSAC = CGAL::Shape_detection::Efficient_RANSAC<RANSACTraits>;
using ShapeDetPlane = CGAL::Shape_detection::Plane<RANSACTraits>;
using ShapeDetCyl = CGAL::Shape_detection::Cylinder<RANSACTraits>;

using PointSet = CGAL::Point_set_3<Point_3>;

using Concurrency_tag = CGAL::Parallel_tag;

// Define how a color should be stored
namespace CGAL {
    template< class F >
    struct Output_rep< ::RGBA, F > {
    const ::RGBA& c;
    static const bool is_specialized = true;
    Output_rep (const ::RGBA& c) : c(c)
    { }
    std::ostream& operator() (std::ostream& out) const
    {
        if (IO::is_ascii(out))
            out << int(c[0]) << " " << int(c[1]) << " " << int(c[2]);
        else {
            // TODO :> this is broken
            //out.write((char*) c, sizeof(uint16_t)*3);
        }
        return out;
    }
    };
} // namespace CGAL

// instance of std::function<bool(double)>
struct Progress_to_std_cerr_callback
{
  mutable std::size_t nb;
  CGAL::Real_timer timer;
  double t_start;
  mutable double t_latest;
  const std::string name;
  Progress_to_std_cerr_callback (const char* name)
    : name (name)
  {
    timer.start();
    t_start = timer.time();
    t_latest = t_start;
  }
  bool operator()(double advancement) const
  {
    // Avoid calling time() at every single iteration, which could
    // impact performances very badly
    ++ nb;
    if (advancement != 1 && nb % 100 != 0)
      return true;
    double t = timer.time();
    if (advancement == 1 || (t - t_latest) > 0.1) // Update every 1/10th of second
    {
      std::cerr << "\r" // Return at the beginning of same line and overwrite
                << name << ": " << int(advancement * 100) << "%";
      if (advancement == 1)
        std::cerr << std::endl;
      t_latest = t;
    }
    return true;
  }
};

/**
 *  Use canopy clustering approach to estimate the number of clusters, K, within a point set
 * According to https://en.wikipedia.org/wiki/Canopy_clustering_algorithm
 *  
 *      NOTE :> don't use this, it's complete slow shit, but contains some useful 
 *          methods in using iterators and vectors that I didn't know about before
 */
size_t canopy_cluster_estimate_k(double t1_loose, double t2_tight, std::vector<PointData> points, boost::shared_ptr<EfficientRANSAC::Shape> shape) {
    size_t k_clusters = 0;
    std::vector<Point_3> canopy; 

    // spatially index and sort the input points since we'll be dropping previous canopies
    CGAL::Spatial_sort_traits_adapter_3<Kernel,PointMap> sort_traits;
    CGAL::spatial_sort<CGAL::Parallel_tag>(
        points.begin(),
        points.end(),
        sort_traits
    );

    while (!points.empty()) {
        Point_3 current = boost::get<0>(points.back());
        canopy = { current };
        k_clusters++;
        points.pop_back();
        std::cout << "Found a cluster candidate." << std::endl;

        auto it = points.begin();
        while (it != points.end()) {
            Point_3 to_test = boost::get<0>(*it);
            double dist = std::sqrt(CGAL::squared_distance(current, to_test));

            if (dist <= t1_loose) {
                // add to canopy if within "loose" distance
                canopy.push_back(to_test);
            }

            if (dist <= t2_tight) {
                it = points.erase(it); // if within tight distance, drop it from set
                //std::cout << "Added to canopy..." << std::endl;
            } else {
                it++;
            }
        }
    }

    return k_clusters;
}

/**
 *  TODO :> docs
 */
int main (int argc, char* argv[]) {

    //
    // Load a .PTS file
    //

    const char* fname = (argc > 1) ? argv[1] : "data/cuboid.pts";
    std::ifstream infile(fname);

    //
    // Process into a Point Set
    //

    std::cout.precision(5);

    std::string line;
    uint64_t total_pts;

    //Point_Set points;
    std::vector<PointData> points;

    // Read first line with point number total

    std::getline(infile, line);
    std::istringstream tokenizer(line);

    if (!(tokenizer >> total_pts)) {
        std::cout << "Failed to read first line count" << std::endl;
        return EXIT_FAILURE;
    }

    // Process remaining lines in file

    for (size_t i = 0; std::getline(infile, line); i++) 
    {
        std::istringstream tokenizer(line);

        double northing, easting, altitude;
        int64_t intensity;
        uint16_t r,g,b;     // read uint16_t bc Unicode, even tho its 1-byte val

        if (!(tokenizer >> northing >> easting >> altitude >> intensity >> r >> g >> b)) {
            std::cout << "Failed to read a line correctly" << std::endl;
            //return EXIT_FAILURE;
            break;
        }

        // Transform from Canonical to Cartesian Grid
        Point_3 point_xyz = Point_3(easting, altitude, northing);
        RGBA rgba_info = {r, g, b};
        RGBA shape_info = {0, 0, 0};
        size_t patch_id = 0;
        RGBA patch_info = {0, 0, 0};
        PointData point = PointData(
            point_xyz, 
            Vector_3(1.0, 1.0, 1.0), 
            rgba_info, 
            shape_info,
            patch_id,
            patch_info,
            i
        );

        points.push_back(point);
    }

    std::cout << "Total Points: " << total_pts << std::endl;

    //
    // grid simplify to minimum resolution (error of scanner)
    //  NOTE :> have to use this instead of WLOP simplify+denoise+regularize,
    //          it takes way too long
    //  NOTE :> didn't test WLOP with Hilbert sorting first, though. 
    //

    double cell_size = 0.003;
    std::vector<PointData>::iterator to_remove;
    to_remove = CGAL::grid_simplify_point_set(
        points,
        cell_size,
        CGAL::parameters::
            point_map( PointMap() ).
            callback(
                Progress_to_std_cerr_callback("Grid Simplification")
            )
    );

    int nb_removed = std::distance(to_remove, points.end());
    std::cerr << "Keep " << total_pts - nb_removed <<
                 " of " << total_pts << " points." << 
                 " (" << 100.0 * (float(total_pts - nb_removed) / float(total_pts)) << "%)" << 
                 std::endl;

    points.erase(to_remove, points.end());          // erase simplified points
    std::vector<PointData>(points).swap(points);    // trim excess capacity

    //
    //  Hilbert/Spatial sort for speedup on incremental/neighbor based CGAL algos
    //      TODO :> Hold off on using sorting until we've trimmed the point set and
    //      oriented normals?  <-- because these operations reorder the set

    CGAL::Spatial_sort_traits_adapter_3<Kernel,PointMap> sort_traits;
    CGAL::spatial_sort<CGAL::Parallel_tag>(
        points.begin(),
        points.end(),
        sort_traits
    );

    // reset the index counters for all the Point_3s within the spatially-sorted container
    for (size_t i=0; i<points.size(); i++) {
        boost::get<6>(points[i]) = i;
    }

    //
    //  Estimate average number of neighbors in local neighborhood for normals 
    //  and planar surface algorithms 
    //      TODO :> ^^^^^

    int nb_neighbors = 32;

    //
    // estimate normal vector orientations
    // 
    CGAL::jet_smooth_point_set<CGAL::Parallel_tag>(
        points,
        nb_neighbors,
        CGAL::parameters::
            point_map(PointMap()).
            degree_fitting(4).
            callback(
                Progress_to_std_cerr_callback("Jet Smoothing - Quadric")
            )
    );

    // use jet-surface fitting
    CGAL::jet_estimate_normals<CGAL::Parallel_tag>(
        points,
        nb_neighbors,
        CGAL::parameters::
            point_map(PointMap()).
            normal_map(NormalMap()).
            callback(
                Progress_to_std_cerr_callback("Jet Estimation - Normals")
            )
    );

    /*
    // orient the normals 
    CGAL::mst_orient_normals(
        points,
        nb_neighbors,
        CGAL::parameters::
            point_map(CGAL::Nth_of_tuple_property_map<0, PointData>()).
            normal_map(CGAL::Nth_of_tuple_property_map<1, PointData>())
    );
    */

    //
    // Write diagnostic PLY
    //

    std::ofstream ofile("diagnostic.ply"); 
    CGAL::IO::set_ascii_mode(ofile); // NOTE :> writing binary of colors is broken
    CGAL::IO::write_PLY_with_properties(
        ofile,
        points,
        CGAL::make_ply_point_writer(
            PointMap()
        ),
        CGAL::make_ply_normal_writer(
            NormalMap()
        ),
        std::make_tuple(
            PhotoColorMap(),
            CGAL::IO::PLY_property<uint8_t>("red"),
            CGAL::IO::PLY_property<uint8_t>("green"),
            CGAL::IO::PLY_property<uint8_t>("blue")
        )
    );

    //
    // attempt planar estimation across the whole shabangalang
    //

    EfficientRANSAC ransac; 
    ransac.set_input(points);
    
    ransac.add_shape_factory<ShapeDetPlane>();
    ransac.add_shape_factory<ShapeDetCyl>();

    EfficientRANSAC::Parameters params;
    // Set probability to miss the largest primitive at each iteration.
    params.probability = 0.05;
    // Detect shapes with at least X points.
    params.min_points = 200;
    // Set maximum Euclidean distance between a point and a shape.
    params.epsilon = 0.1;
    // Set maximum Euclidean distance (neighbor dist) between points to be clustered.
    params.cluster_epsilon = 0.3;
    // Set maximum normal deviation.
    // 0.9 < dot(surface_normal, point_normal);
    params.normal_threshold = 0.9;

    ransac.detect(params);

    double coverage = double(points.size() - ransac.number_of_unassigned_points()) / double(points.size());
    std::cout << std::distance(ransac.shapes().end(), ransac.shapes().begin()) <<
    " shape primitives detected. (" << 100* coverage << "% Coverage)" << std::endl;

    // iterate through shapes 
    EfficientRANSAC::Shape_range shapes = ransac.shapes();
    EfficientRANSAC::Shape_range::iterator it = shapes.begin();
    srand(time(0));

    std::cout << "Shape primitives: " << std::endl;
    while (it != shapes.end()) {
        boost::shared_ptr<EfficientRANSAC::Shape> shape = *it;
        std::cout << (*it)->info() << std::endl;

        // generate a random color code for this shape
        RGBA rgb;
        for (int i=0; i<3; i++) {
            rgb[i] = rand()%256;
        }

        // Form triangulation to later convert into Graph representation
        using VertexInfoBase = CGAL::Triangulation_vertex_base_with_info_3<
                                    PointData,
                                    Kernel
                                >;
        using TriTraits = CGAL::Triangulation_data_structure_3<
                                VertexInfoBase,
                                CGAL::Delaunay_triangulation_cell_base_3<Kernel>,
                                CGAL::Parallel_tag
                            >;
        using Triangulation_3 = CGAL::Delaunay_triangulation_3<Kernel, TriTraits>;

        Triangulation_3 tr;

        // Iterate through point indices assigned to each detected shape. 
        std::vector<std::size_t>::const_iterator 
            index_it = (*it)->indices_of_assigned_points().begin();

        while (index_it != (*it)->indices_of_assigned_points().end()) {
            PointData& p = *(points.begin() + (*index_it));

            // assign shape diagnostic color info
            boost::get<3>(p) = rgb;

            // insert Point_3 data for triangulation and attach PointData info
            TriTraits::Vertex_handle vertex = tr.insert(boost::get<0>(p));
            vertex->info() = p;

            index_it++; // next assigned point
        }
        
        std::cout << "Found triangulation with: \n\t" << 
            tr.number_of_vertices() << "\tvertices\n\t" <<
            tr.number_of_edges() << "\tedges\n\t" <<
            tr.number_of_facets() << "\tfacets" << std::endl;

        // build a Graph out of the triangulation that we can do a Minimum-Spanning-Tree on
        // examples taken from https://www.boost.org/doc/libs/1_80_0/libs/graph/example/kruskal-example.cpp
        using Graph = boost::adjacency_list<
                        boost::vecS,            // OutEdgeList
                        boost::vecS,            // VertexList
                        boost::undirectedS,     // Directed
                        boost::no_property,     // VertexProperties
                        boost::property< boost::edge_weight_t, double >  // EdgeProperties
                        >;
        using Edge = boost::graph_traits<Graph>::edge_descriptor;
        using E = std::pair< size_t, size_t >; // <: TODO - should be iterator index of vertex in Triangulation_3 instead of size_t?

        std::vector<E> edges;
        std::vector<double> weights;

        // iterate over Finite Facets in the triangulation to access the Geom_traits::Triangle
        // add vertices and their "neighbors" for each facet to the Graph representation
        for (TriTraits::Vertex_handle vtx : tr.finite_vertex_handles()) {
            std::vector<TriTraits::Vertex_handle> adjacent;
            auto shit = tr.adjacent_vertices(vtx, std::back_inserter(adjacent));

            auto vertex = tr.point(vtx);

            for (auto adjacent_vtx : adjacent) {
                auto next_vtx = tr.point(adjacent_vtx);
                double weight = std::sqrt(CGAL::squared_distance(vertex, next_vtx));

                /*
                E edge = E(
                        boost::get<6>(vtx->info()),
                        boost::get<6>(adjacent_vtx->info())
                    );
                */

                edges.push_back(
                    E(
                        boost::get<6>(vtx->info()),
                        boost::get<6>(adjacent_vtx->info())
                    )
                );
                weights.push_back(weight);

                //std::cout << "Edge " << edge.first << " <--> " << edge.second << " with weight " << weight << std::endl;
            }
        }

        std::cout << "copy out edge and weight vectors to arrays" << std::endl;

        // build Graph from `edge_array` and `weights`
        E edge_array[] = { E(0, 2), E(3,1), E(1, 3), E(1, 4), E(2, 1), E(2, 3), E(3, 4), E(4, 0), E(4, 1) };
        double weight_array[] = { 1.0, 1.0, 1.0, 2.0, 7.0, 3.0, 1.0, 1.0, 1.0 };

        /*
        //size_t length = edges.size();
        E* edge_array = new E[ edges.size() ];
        std::copy(edges.begin(), edges.end(), edge_array);

        //double weight_array[ weights.size() ];
        double* weight_array = new double[ weights.size() ];
        std::copy(weights.begin(), weights.end(), weight_array);
        */

        int num_nodes = tr.number_of_vertices();
        std::size_t num_edges = sizeof(edge_array) / sizeof(E);

        Graph g(edge_array, edge_array + num_edges, weight_array, num_nodes);

        // build Euclidean-Minimum-Spanning-Tree (EMST) as list of simplex edges between vertices
        boost::property_map<Graph, boost::edge_weight_t>::type weight = boost::get(boost::edge_weight, g);
        std::vector<Edge> spanning_tree;

        boost::kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

        // - traverse EMST from start of list, performing "cuts" into "patches" when we have hit
        // max patch distance (euclidean) from current "first" vertex of "patch". 
        // - have to be able to access Triangulation_3 vertex info (via `locate`?) here
        // - foreach collection of PointData in patch, assign `patch_id` and diagnostic color info,
        //   then commit individual "patches" collections of Point_3 and RGBA photocolor to DB 
        std::cout << "Found minimum spanning tree of " << spanning_tree.size() << " edges for #vertices " << tr.number_of_vertices() << std::endl;
        std::cout << "Print the edges in the MST:" << std::endl;
        for (std::vector< Edge >::iterator ei = spanning_tree.begin();
            ei != spanning_tree.end(); ++ei)
        {
            std::cout << boost::source(*ei, g) << " <--> " << boost::target(*ei, g)
                    << " with weight of " << weight[*ei] << std::endl;
        }


        //delete[] edge_array;
        //delete[] weight_array;
        it++; // next shape
    }

    //
    //  Diagnostic color output of detected shapes
    //
    std::ofstream shapefile("diagnostic-RANSAC-shapes.ply");
    CGAL::IO::set_ascii_mode(ofile); // NOTE :> writing binary of colors is broken
    CGAL::IO::write_PLY_with_properties(
        shapefile,
        points,
        CGAL::make_ply_point_writer(
            PointMap()
        ),
        CGAL::make_ply_normal_writer(
            NormalMap()
        ),
        std::make_tuple(
            ShapeColorMap(),
            CGAL::IO::PLY_property<uint8_t>("red"),
            CGAL::IO::PLY_property<uint8_t>("green"),
            CGAL::IO::PLY_property<uint8_t>("blue")
        )
    );

    return EXIT_SUCCESS;
}

