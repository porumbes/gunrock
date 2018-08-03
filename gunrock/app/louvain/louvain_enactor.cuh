// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * louvain_enactor.cuh
 *
 * @brief Louvain Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/louvain/louvain_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace louvain {

/**
 * @brief Speciflying parameters for Louvain Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    GUARD_CU(app::UseParameters_enactor(parameters));

    GUARD_CU(parameters.Use<uint64_t>(
        "max-passes",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        10,
        "Maximum number of passes to run the louvain algorithm.",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<uint64_t>(
        "max-iters",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        10,
        "Maximum number of iterations to run for each pass.",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<double>(
        "pass-th",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        1e-4,
        "Modularity threshold to continue further passes.",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<double>(
        "iter-th",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        1e-6,
        "Modularity threshold to continue further iterations within a pass.",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<double>(
        "1st-th",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        1e-4,
        "Modularity threshold to continue further iterations in the first pass.",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<bool>(
        "pass-stats",
        util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        false,
        "Whether to show per-pass stats.",
        __FILE__, __LINE__));

   GUARD_CU(parameters.Use<bool>(
        "iter-stats",
        util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        false,
        "Whether to show per-iteration stats.",
        __FILE__, __LINE__));

    return retval;
}

/**
 * @brief defination of SSSP iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct LouvainIterationLoop : public IterationLoopBase
    <EnactorT, Use_FullQ | Push
    // TODO: if needed, stack more option, e.g.:
    // | (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
    // Update_Predecessors : 0x0)
    >
{
    typedef typename EnactorT::VertexT VertexT;
    typedef typename EnactorT::SizeT   SizeT;
    // TODO: make alias of data types used in the enactor, e.g.:
    typedef typename EnactorT::ValueT  ValueT;

    // TODO: make alias of graph representation used in the enactor, e.g.:
    typedef typename EnactorT::Problem::GraphT::CsrT CsrT;

    typedef typename EnactorT::Problem::GraphT::GpT  GpT;
    typedef IterationLoopBase
        <EnactorT, Use_FullQ | Push
        // TODO: add the same options as in template parameters here, e.g.:
        // | (((EnactorT::Problem::FLAG & Mark_Predecessors) != 0) ?
        // Update_Predecessors : 0x0)
        > BaseIterationLoop;

    LouvainIterationLoop() : BaseIterationLoop() {}

    /**
     * @brief Core computation of sssp, one iteration
     * @param[in] peer_ Which GPU peers to work on, 0 means local
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Core(int peer_ = 0)
    {
        // Data alias the enactor works on
        auto         &data_slice         =   this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        auto         &enactor_slice      =   this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];
        auto         &enactor_stats      =   enactor_slice.enactor_stats;
        auto         &graph              =   data_slice.sub_graph[0];
        auto         &frontier           =   enactor_slice.frontier;
        auto         &oprtr_parameters   =   enactor_slice.oprtr_parameters;
        auto         &retval             =   enactor_stats.retval;
        auto         &pass_num           =   enactor_stats.iteration;
        auto         &weights            =   graph.CsrT::edge_values;
        auto         &w_v2               =   data_slice.w_v2;
        auto         &w_v2self           =   data_slice.w_v2slef;
        cudaStream_t  stream             =   oprtr_parameters.stream;
        auto          target             =   util::DEVICE;
        VertexT       iter               =   0;
        util::Array1D<SizeT, VertexT>* null_frontier = NULL;

        // Pass initialization
        GUARD_CU(w_v2.ForAll([w_v2self, current_communities]
            (ValueT *w_v2_, const SizeT &v)
            {
                w_v2_    [v] = 0;
                w_v2self [v] = 0;
                current_communities[v] = v;
            }, graph.nodes, target, stream));
        //GUARD_CU(w_v2self.ForEach([](ValueT &w)
        //    {
        //        w = 0;
        //    }, graph.nodes, util::DEVICE, oprtr_parameters.stream));

        // Accumulate edge values
        auto accu_op = [w_v2, w_v2self, weights] __host__ __device__ (
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool
        {
            atomicAdd(w_v2 + src, weights[edge_id]);
            if (src == dest)
                atomicAdd(w_v2self + src, weights[edge_id]);
            return false;
        };
        frontier.queue_length = graph.nodes;
        frontier.queue_reset  = true;
        oprtr_parameters.advance_mode = "ALL_EDGES";
        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), null_frontier, null_frontier,
            oprtr_parameters, accu_op));

        GUARD_CU(w_c2.ForEach(w_v2, [](ValueT &w_c, const ValueT &w_v)
            {
                w_c = w_v;
            }, graph.nodes, target, stream));

        int iter_num = 0;
        ValueT pass_gain = 0;
        ValueT iter_gain = 0;
        bool to_continue = true;

        // Iterations
        while (to_continue)
        {
            if (iter_stats)
                iter_timer.Start();
            iter_gain = 0;

            GUARD_CU(edge_comms0.ForAll(
                [edge_weights0, weights, current_communities, column_indices]
                (VertexT *e_comms, const SizeT &e)
                {
                    e_comms[e] = current_communities[column_indices[e]];
                    edge_weights0[e] = weights[e];
                }, graph.edges, target, stream));

            cub::DoubleBuffer<VertexT> keys_frontiers(
                edge_comms0.GetPointer(util::DEVICE),
                edge_comms1.GetPointer(util::DEVICE));
            cub::DoubleBuffer<ValueT > vals_frontiers(
                edge_weights0.GetPointer(util::DEVICE),
                edge_weights1.GetPointer(util::DEVICE));

            GUARD_CU(cub_SegmentedRadixSortPairs(
                cub_temp_space,
                keys_frontiers,
                vals_frontiers,
                graph.edges, graph.nodes,
                graph.CsrT::row_offsets.GetPointer(util::DEVICE),
                graph.CsrT::row_offsets.GetPointer(util::DEVICE) + (graph.nodes + 1),
                0, std::ceil(std::log(graph.nodes)), stream));

            GUARD_CU(temp_seg_offsets.ForAll(
                [](SizeT *offsets, const SizeT &e){
                    offsets[e] = e;
                }, graph.edges, target, stream));

            // Filter in order
            auto edge_comms = keys_frontiers.Current();
            GUARD_CU(cub_SelectIf(
                cub_temp_space,
                temp_seg_offsets.GetPointer(util::DEVICE),
                neighbor_comm_offsets.GetPointer(util::DEVICE),
                num_neighbor_comms.GetPointer(util::DEVICE),
                graph.edges,
                [edge_comms](const SizeT &e){
                    if (e == 0)
                        return true;
                    if (edge_comms[e] != edge_comms[e-1])
                        return true;
                    SizeT pos = util::BinarySearch(row_offsets, e, 0, edges);
                    if (row_offsets[pos] == e)
                        return true;
                    return false;
                }, stream));

            // TODO: use a special kernel to find community boundary for vertices

            GUARD_CU(oprtr::Set(neighbor_comm_offsets.GetPointer(util::DEVICE)
                + (num_neighbor_comms[0] + 1), graph.edges, target, stream));
            GUARD_CU(num_neighbor_comms.Move(util::DEVICE, util::HOST, 1, 0, stream));

            GUARD_CU2(cudaStreamSynchronize(stream),
                "cudaStreamSynchronize failed.");

            GUARD_CU(cub_SegmentedSum(
                cub_temp_space,
                vals_frontiers.Current(),
                vals_frontiers.Alternative(),
                num_neighbor_comms[0],
                neighbor_comm_offsets.GetPointer(util::DEVICE),
                neighbor_comm_offsets.GetPointer(util::DEVICE)
                    + (num_neighbor_comms[0] + 1), stream));

            GUARD_CU(gain_bases.ForAll(
                [w_v2, current_communities, w_v2self, w_v2c_org, m2]
                (ValueT *gains, const VertexT &v)
                {
                    ValueT w_v2_v = w_v2[v];
                    VertexT comm = current_communities[v];
                    gains[v] = w_v2self[v] - w_v2c_org[v]
                        - (w_v2_v - w_c2[comm]) * w_v2_v / m2;
                }, graphs.nodes, target, stream));

            ValueT *w_v2c = vals_frontiers.Alternative();
            GUARD_CU(oprtr::ForAll(
                vals_frontiers.Current(),
                [vc_offsets, w_v2c, gain_bases, w_c2, w_v2, m2]
                (ValueT *gains, const SizeT &pos)
                {
                    VertexT v = BinarySearch(
                        vc_offsets + 0, pos, 0, num_neighbor_comms_);
                    gains[pos] = gain_bases[v] + w_v2c[pos]
                        - w_c2[comm] * w_v2[v] / m2;
                }, num_neighbor_comms[0], target, stream));



            iter_gain *= 2;
            iter_gain /= m2;
            q += iter_gain;
            pass_gain += iter_gain;
            if (iter_stats)
            {
                iter_timer.Stop();
                util::PrintMsg("pass " + std::to_string(pass_num)
                    + ", iter " + std::to_string(iter_num)
                    + ", q = " + std::to_string(q)
                    + ", iter_gain = " + std::to_string(iter_gain)
                    + ", pass_gain = " + std::to_string(pass_gain)
                    + ", elapsed = "
                    + std::to_string(iter_timer.ElapsedMillis()), iter_stats);
            }
            iter_num ++;
            if ((pass_num != 0 && iter_gain < iter_gain_threshold) ||
                (pass_num == 0 && iter_gain < first_threshold) ||
                iter_num >= max_iters)
                to_continue = false;
        }

        // The filter operation
        auto filter_op = [
        // TODO: if needed, pass data used by the lambda
        ] __host__ __device__ (
            const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool
        {
            // TODO: finll in the per-vertex filter operation, e.g.:
            // if (!util::isValid(dest)) return false;
            return true;
        };

        // Call the advance operator, using the advance operation
        // TODO: modify the operator callers according to algorithmic needs,
        //       this example only uses an advance + a filter, with
        //       possible optimization to fuze the two kernels.
        //       Define more operations (i.e. lambdas) if needed
        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), frontier.V_Q(), frontier.Next_V_Q(),
            oprtr_parameters, advance_op, filter_op));

        if (oprtr_parameters.advance_mode != "LB_CULL" &&
            oprtr_parameters.advance_mode != "LB_LIGHT_CULL")
        {
            frontier.queue_reset = false;
            // Call the filter operator, using the filter operation
            GUARD_CU(oprtr::Filter<oprtr::OprtrType_V2V>(
                graph.csr(), frontier.V_Q(), frontier.Next_V_Q(),
                oprtr_parameters, filter_op));
        }

        // Get back the resulted frontier length
        GUARD_CU(frontier.work_progress.GetQueueLength(
            frontier.queue_index, frontier.queue_length,
            false, oprtr_parameters.stream, true));

        return retval;
    }

    /**
     * @brief Routine to combine received data and local data
     * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each transmition item, typed VertexT
     * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each transmition item, typed ValueT
     * @param  received_length The numver of transmition items received
     * @param[in] peer_ which peer GPU the data came from
     * \return cudaError_t error message(s), if any
     */
    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    cudaError_t ExpandIncoming(SizeT &received_length, int peer_)
    {
        auto         &data_slice         =   this -> enactor ->
            problem -> data_slices[this -> gpu_num][0];
        auto         &enactor_slice      =   this -> enactor ->
            enactor_slices[this -> gpu_num * this -> enactor -> num_gpus + peer_];
        //auto iteration = enactor_slice.enactor_stats.iteration;
        // TODO: add problem specific data alias here, e.g.:
        // auto         &distances          =   data_slice.distances;

        auto expand_op = [
        // TODO: pass data used by the lambda, e.g.:
        // distances
        ] __host__ __device__(
            VertexT &key, const SizeT &in_pos,
            VertexT *vertex_associate_ins,
            ValueT  *value__associate_ins) -> bool
        {
            // TODO: fill in the lambda to combine received and local data, e.g.:
            // ValueT in_val  = value__associate_ins[in_pos];
            // ValueT old_val = atomicMin(distances + key, in_val);
            // if (old_val <= in_val)
            //     return false;
            return true;
        };

        cudaError_t retval = LouvainIterationLoop:: template ExpandIncomingBase
            <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            (received_length, peer_, expand_op);
        return retval;
    }
}; // end of SSSPIteration

/**
 * @brief Louvain enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <
    typename _Problem,
    util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor :
    public EnactorBase<
        typename _Problem::GraphT,
        typename _Problem::GraphT::VertexT, // TODO: change to other label types used for the operators, e.g.: typename _Problem::LabelT,
        typename _Problem::GraphT::ValueT, // TODO: change to other value types used for inter GPU communication, e.g.: typename _Problem::ValueT,
        ARRAY_FLAG, cudaHostRegisterFlag>
{
public:
    typedef _Problem                   Problem ;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexT  VertexT ;
    typedef typename Problem::GraphT   GraphT  ;
    // TODO: change according to the EnactorBase template parameters above
    typedef typename GraphT::VertexT   LabelT  ; // e.g. typedef typename Problem::LabelT LabelT;
    typedef typename GraphT::ValueT    ValueT  ; // e.g. typedef typename Problem::ValueT ValueT;
    typedef EnactorBase<GraphT , LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
        BaseEnactor;
    typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag>
        EnactorT;
    typedef LouvainIterationLoop<EnactorT> IterationT;

    Problem     *problem   ;
    IterationT  *iterations;

    /**
     * @brief SSSPEnactor constructor
     */
    Enactor() :
        BaseEnactor("Louvain"),
        problem    (NULL  )
    {
        // TODO: change according to algorithmic needs
        this -> max_num_vertex_associates = 0;
        this -> max_num_value__associates = 1;
    }

    /**
     * @brief SSSPEnactor destructor
     */
    virtual ~Enactor()
    {
        //Release();
    }

    /*
     * @brief Releasing allocated memory space
     * @param target The location to release memory from
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseEnactor::Release(target));
        delete []iterations; iterations = NULL;
        problem = NULL;
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Initialize the problem.
     * @param[in] parameters Running parameters.
     * @param[in] problem The problem object.
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Init(
        //util::Parameters &parameters,
        Problem          &problem,
        util::Location    target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        this->problem = &problem;

        // Lazy initialization
        GUARD_CU(BaseEnactor::Init(
            problem, Enactor_None,
            // TODO: change to how many frontier queues, and their types
            2, NULL,
            target, false));
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++)
        {
            GUARD_CU(util::SetDevice(this -> gpu_idx[gpu]));
            auto &enactor_slice
                = this -> enactor_slices[gpu * this -> num_gpus + 0];
            auto &graph = problem.sub_graphs[gpu];
            GUARD_CU(enactor_slice.frontier.Allocate(
                graph.nodes, graph.edges, this -> queue_factors));
        }

        iterations = new IterationT[this -> num_gpus];
        for (int gpu = 0; gpu < this -> num_gpus; gpu ++)
        {
            GUARD_CU(iterations[gpu].Init(this, gpu));
        }

        GUARD_CU(this -> Init_Threads(this,
            (CUT_THREADROUTINE)&(GunrockThread<EnactorT>)));
        return retval;
    }

    /**
      * @brief one run of sssp, to be called within GunrockThread
      * @param thread_data Data for the CPU thread
      * \return cudaError_t error message(s), if any
      */
    cudaError_t Run(ThreadSlice &thread_data)
    {
        gunrock::app::Iteration_Loop<
            // TODO: change to how many {VertexT, ValueT} data need to communicate
            //       per element in the inter-GPU sub-frontiers
            0, 1,
            IterationT>(
            thread_data, iterations[thread_data.thread_num]);
        return cudaSuccess;
    }

    /**
     * @brief Reset enactor
     * @param[in] src Source node to start primitive.
     * @param[in] target Target location of data
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Reset(
        // TODO: add problem specific info, e.g.:
        // VertexT src,
        util::Location target = util::DEVICE)
    {
        typedef typename GraphT::GpT GpT;
        cudaError_t retval = cudaSuccess;
        GUARD_CU(BaseEnactor::Reset(target));

        // TODO: Initialize frontiers according to the algorithm, e.g.:
        for (int gpu = 0; gpu < this->num_gpus; gpu++)
        {
        //    if ((this->num_gpus == 1) ||
        //         (gpu == this->problem->org_graph->GpT::partition_table[src]))
        //    {
        //        this -> thread_slices[gpu].init_size = 1;
        //        for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
        //        {
        //            auto &frontier = this ->
        //                enactor_slices[gpu * this -> num_gpus + peer_].frontier;
        //            frontier.queue_length = (peer_ == 0) ? 1 : 0;
        //            if (peer_ == 0)
        //            {
        //                GUARD_CU(frontier.V_Q() -> ForEach(
        //                    [src]__host__ __device__ (VertexT &v)
        //                {
        //                    v = src;
        //                }, 1, target, 0));
        //            }
        //        }
        //    }
        //
        //    else {
                this -> thread_slices[gpu].init_size = 0;
                for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
                {
                    this -> enactor_slices[gpu * this -> num_gpus + peer_]
                        .frontier.queue_length = 0;
                }
        //    }
        }
        GUARD_CU(BaseEnactor::Sync());
        return retval;
    }

    /**
     * @brief Enacts a SSSP computing on the specified graph.
     * @param[in] src Source node to start primitive.
     * \return cudaError_t error message(s), if any
     */
    cudaError_t Enact()
    {
        cudaError_t  retval     = cudaSuccess;
        GUARD_CU(this -> Run_Threads(this));
        util::PrintMsg("GPU Louvain Done.", this -> flag & Debug);
        return retval;
    }

    /** @} */
};

} // namespace louvain
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: