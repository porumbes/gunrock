// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cc_problem.cuh
 *
 * @brief GPU Storage management Structure for Connected Components (CC) Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace cc {
// </TODO>


/**
 * @brief Speciflying parameters for hello Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(
    util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    GUARD_CU(gunrock::app::UseParameters_problem(parameters));

    // <TODO> Add problem specific command-line parameter usages here, e.g.:
    // GUARD_CU(parameters.Use<bool>(
    //    "mark-pred",
    //    util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
    //    false,
    //    "Whether to mark predecessor info.",
    //    __FILE__, __LINE__));
    // </TODO>

    return retval;
}

/**
 * @brief Template Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _FLAG    Problem flags
 */
template <
    typename _GraphT,
    ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG>
{
    typedef _GraphT GraphT;
    static const ProblemFlag FLAG = _FLAG;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::ValueT  ValueT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::CsrT    CsrT;
    typedef typename GraphT::GpT     GpT;

    typedef ProblemBase   <GraphT, FLAG> BaseProblem;
    typedef DataSliceBase <GraphT, FLAG> BaseDataSlice;

    // ----------------------------------------------------------------
    // Dataslice structure

    /**
     * @brief Data structure containing problem specific data on indivual GPU.
     */
    struct DataSlice : BaseDataSlice
    {
        // device storage arrays
        util::Array1D<SizeT, VertexT> component_ids; /**< Used for component id */
        util::Array1D<SizeT, signed char> masks;         /**< Size equals to node number, show if a node is the root */
        util::Array1D<SizeT, bool    > marks;         /**< Size equals to edge number, show if two vertices belong to the same component */
        util::Array1D<SizeT, VertexT> froms;         /**< Size equals to edge number, from vertex of one edge */
        util::Array1D<SizeT, VertexT> tos;           /**< Size equals to edge number, to vertex of one edge */
        util::Array1D<SizeT, int     > vertex_flag;   /**< Finish flag for per-vertex kernels in CC algorithm */
        util::Array1D<SizeT, int     > edge_flag;     /**< Finish flag for per-edge kernels in CC algorithm */
        util::Array1D<SizeT, VertexT*> vertex_associate_ins;
        int turn;
        bool has_change, previous_change;
        bool scanned_queue_computed;
        VertexT *temp_vertex_out;
        VertexT *temp_comp_out;
        // </TODO>

        /*
         * @brief Default constructor
         */
        DataSlice() : BaseDataSlice()
        {
            component_ids.SetName("component_ids");
            masks        .SetName("masks"        );
            marks        .SetName("marks"        );
            froms        .SetName("froms"        );
            tos          .SetName("tos"          );
            vertex_flag  .SetName("vertex_flag"  );
            edge_flag    .SetName("edge_flag"    );
            vertex_associate_ins.SetName("vertex_associate_ins");
            turn          = 0;
            has_change    = true;
            previous_change = true;
            scanned_queue_computed = false;
            temp_vertex_out = NULL;
            temp_comp_out = NULL;
            // </TODO>
        }

        /*
         * @brief Default destructor
         */
        virtual ~DataSlice() { Release(); }

        /*
         * @brief Releasing allocated memory space
         * @param[in] target      The location to release memory from
         * \return    cudaError_t Error message(s), if any
         */
        cudaError_t Release(util::Location target = util::LOCATION_ALL)
        {
            cudaError_t retval = cudaSuccess;
            if (target & util::DEVICE)
                GUARD_CU(util::SetDevice(this->gpu_idx));

            component_ids.Release();
            masks        .Release();
            marks        .Release();
            froms        .Release();
            tos          .Release();
            vertex_flag  .Release();
            edge_flag    .Release();
            vertex_associate_ins.Release();
            // </TODO>

            GUARD_CU(BaseDataSlice::Release(target));
            return retval;
        }

        /**
         * @brief initialize CC-specific data on each gpu
         * @param     sub_graph   Sub graph on the GPU.
         * @param[in] gpu_idx     GPU device index
         * @param[in] target      Targeting device location
         * @param[in] flag        Problem flag containling options
         * \return    cudaError_t Error message(s), if any
         */
        cudaError_t Init(
            GraphT        &sub_graph,
            int            num_gpus,    // SDP can we ditch at have a different Init for num_gpus > 1?
            int            gpu_idx,
            util::Location target,      // SDP is this parameter really needed?
            ProblemFlag    flag)
        {
            cudaError_t retval  = cudaSuccess;

            // SDP Move the CSR graph bits to the GPU. The 'tos' needs
            // to reference the 'column_indices' on DEVICE
            GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));
            if(target & util::DEVICE)
            {
                GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this->stream));
            }

	        GUARD_CU(component_ids         .Allocate(sub_graph.nodes, util::DEVICE));
	        GUARD_CU(masks                 .Allocate(sub_graph.nodes, util::DEVICE));
            GUARD_CU(marks                 .Allocate(sub_graph.edges, util::DEVICE));	        
	        GUARD_CU(vertex_flag           .Allocate(1, util::HOST | util::DEVICE));
	        GUARD_CU(edge_flag             .Allocate(1, util::HOST | util::DEVICE));
	        GUARD_CU(vertex_associate_ins  .Allocate(num_gpus, util::HOST | util::DEVICE));

            //
            // Construct coo from/to edge list from row_offsets and column_indices
            //
            // SDP -- can probably avoid getting COO this way
            GUARD_CU(froms.Allocate(sub_graph.edges, util::HOST | util::DEVICE));
            auto col_idx_ptr = sub_graph.column_indices.GetPointer(util::DEVICE);
            GUARD_CU(tos.SetPointer(col_idx_ptr, sub_graph.edges, util::DEVICE));

            for (VertexT node=0; node < sub_graph.nodes; node++)
            {
                SizeT start_edge = sub_graph.row_offsets[node  ]; // SDP row start
                SizeT end_edge   = sub_graph.row_offsets[node+1]; // SDP see up to this number of items in row

                for (SizeT edge = start_edge; edge < end_edge; ++edge) // SDP walk across the row
                {
                    froms[edge] = node;
                    //tos  [edge] = graph->column_indices[edge];
                }
            }
            GUARD_CU(froms.Move(util::HOST, util::DEVICE));
            GUARD_CU(froms.Release(util::HOST));
            // </TODO>

            return retval;
        }

        /**
         * @brief Reset problem function. Must be called prior to each run.
         * @param[in] target      Targeting device location
         * \return    cudaError_t Error message(s), if any
         */
        cudaError_t Reset(util::Location target = util::DEVICE)
        {
            cudaError_t retval = cudaSuccess;
            SizeT nodes = this -> sub_graph -> nodes;
            SizeT edges = this -> sub_graph -> edges;

            // Ensure data are allocated
            // <TODO> ensure size of problem specific data:
            // SDP looks like we need to "ensure" sizes for 
            // all allocated problem specific data. Not 
            // super cool -- repeating the allocation
            // calls with "EnsureSize"

            GUARD_CU(component_ids         .EnsureSize_(nodes, util::DEVICE));
            GUARD_CU(masks                 .EnsureSize_(nodes, util::DEVICE));
            GUARD_CU(marks                 .EnsureSize_(edges, util::DEVICE));            
            GUARD_CU(vertex_flag           .EnsureSize_(1, util::HOST | util::DEVICE));
            GUARD_CU(edge_flag             .EnsureSize_(1, util::HOST | util::DEVICE));
            GUARD_CU(vertex_associate_ins  .EnsureSize_(1, util::HOST | util::DEVICE));
            // </TODO>

            // Reset data
            // Allocate output component_ids if necessary
            //util::MemsetIdxKernel<<<128, 128>>>(component_ids .GetPointer(util::DEVICE), nodes);
            GUARD_CU(component_ids.ForAll([]__host__ __device__ (VertexT *component_ids_, const SizeT &id){
                    // SDP initialize every node to its own component id
                    component_ids_[id] = id;
                }, nodes, util::DEVICE, this -> stream));

            // Allocate marks if necessary
            //util::MemsetKernel   <<<128, 128>>>(marks         .GetPointer(util::DEVICE), false, edges);
            GUARD_CU(marks.ForEach([]__host__ __device__ (bool &mark){
                mark = false;
            }, edges, util::DEVICE, this -> stream));

            // Allocate masks if necessary
            //util::MemsetKernel    <<<128, 128>>>(masks        .GetPointer(util::DEVICE), (signed char)0, nodes);
            GUARD_CU(masks.ForEach([]__host__ __device__ (signed char &x){
               x = (signed char)0;
            }, nodes, util::DEVICE, this -> stream));

            // Reset stop condition variables
            vertex_flag[0]=0;
            GUARD_CU(vertex_flag.Move(util::HOST, util::DEVICE));
            edge_flag[0]=0;
            GUARD_CU(edge_flag.Move(util::HOST, util::DEVICE));

            // </TODO>

            return retval;
        }
    }; // DataSlice

    // Set of data slices (one for each GPU)
    util::Array1D<SizeT, DataSlice> *data_slices;

    // Members
    SizeT        num_components;

    // ----------------------------------------------------------------
    // Problem Methods

    /**
     * @brief hello default constructor
     */
    Problem(
        util::Parameters &_parameters,
        ProblemFlag _flag = Problem_None) :
        BaseProblem(_parameters, _flag),
        data_slices(NULL) {}

    /**
     * @brief hello default destructor
     */
    virtual ~Problem() { Release(); }

    /*
     * @brief Releasing allocated memory space
     * @param[in] target      The location to release memory from
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;
        if (data_slices == NULL) return retval;
        for (int i = 0; i < this->num_gpus; i++)
            GUARD_CU(data_slices[i].Release(target));

        if ((target & util::HOST) != 0 &&
            data_slices[0].GetPointer(util::DEVICE) == NULL)
        {
            delete[] data_slices; data_slices=NULL;
        }
        GUARD_CU(BaseProblem::Release(target));
        return retval;
    }
    
    /**
     * @brief Copy result component ids computed on the GPU back to a host-side vector.
...  *
     * @param[out] h_component_ids host-side vector to store computed component ids.
     *
     * \return     cudaError_t Error message(s), if any
     */
    cudaError_t Extract(
        VertexT *h_component_ids,
        // </TODO>
        util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        SizeT nodes = this -> org_graph -> nodes;

        // SDP: Assume only one gpu for now
        auto &data_slice = data_slices[0][0];

        // Set device
        if (target == util::DEVICE) {
            GUARD_CU(util::SetDevice(this->gpu_idx[0]));

            GUARD_CU(data_slice.component_ids.SetPointer(h_component_ids,
                                                         nodes,
                                                         util::HOST));
            GUARD_CU(data_slice.component_ids.Move(util::DEVICE, util::HOST));

            // </TODO>
        } else if (target == util::HOST) { // SDP: data is on CPU
            GUARD_CU(data_slice.component_ids.ForEach(h_component_ids,
                []__host__ __device__ (const VertexT &device_component_id, VertexT &host_component_id){
                    host_component_id = device_component_id;
                }, nodes, util::HOST));
            // </TODO>
        }

        // SDP count the number of components

        int *marker = new int[nodes];
        assert(nullptr != marker);
        memset(marker, 0, sizeof(int) * nodes);

        num_components=0;
        for (int node=0; node < nodes; node++) {
            if (marker[h_component_ids[node]] == 0) {
                num_components++;
                //printf("%d\t ",node);
                marker[h_component_ids[node]]=1;
            }
        }

        delete [] marker;

        printf("CC Extract num_components = %d\n", num_components);

        return retval;
    }

    /**
     * @brief initialization function.
     * @param     graph       The graph that SSSP processes on
     * @param[in] Location    Memory location to work on
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(
            GraphT           &graph,
            util::Location    target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseProblem::Init(graph, target));
        data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

        // <TODO> get problem specific flags from parameters, e.g.:
        // if (this -> parameters.template Get<bool>("mark-pred"))
        //    this -> flag = this -> flag | Mark_Predecessors;
        // </TODO>

        for (int gpu = 0; gpu < this->num_gpus; gpu++) {
            data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
            if (target & util::DEVICE)
                GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

            GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

            auto &data_slice = data_slices[gpu][0];
            GUARD_CU(data_slice.Init(
                this -> sub_graphs[gpu],
                this -> num_gpus,
                this -> gpu_idx[gpu],
                target,
                this -> flag
            ));
        }

        return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] src      Source vertex to start.
     * @param[in] location Memory location to work on
     * \return cudaError_t Error message(s), if any
     */
    cudaError_t Reset(
        // <TODO> problem specific data if necessary, eg
        // VertexT src,
        // </TODO>
        util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;

        // Reset data slices
        for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
            if (target & util::DEVICE)
                GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
            GUARD_CU(data_slices[gpu] -> Reset(target));
            GUARD_CU(data_slices[gpu].Move(util::HOST, target));
        }

        // <TODO> Additional problem specific initialization
        // </TODO>

        GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
        return retval;
    }
};

} //namespace Template
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
