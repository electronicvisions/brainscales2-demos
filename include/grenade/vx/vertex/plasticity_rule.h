#pragma once
#include "grenade/vx/connection_type.h"
#include "grenade/vx/event.h"
#include "grenade/vx/port.h"
#include "grenade/vx/vertex/neuron_view.h"
#include "grenade/vx/vertex/plasticity_rule/observable_data_type.h"
#include "halco/common/geometry.h"
#include "halco/common/typed_array.h"
#include "halco/hicann-dls/vx/v3/chip.h"
#include "halco/hicann-dls/vx/v3/synapse.h"
#include "haldls/vx/v3/neuron.h"
#include "hate/visibility.h"
#include <array>
#include <cstddef>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cereal {
struct access;
} // namespace cereal

namespace grenade::vx {

struct Int8;
struct PortRestriction;

namespace vertex {

struct SynapseArrayView;

/**
 * A plasticity rule to operate on synapse array views.
 */
struct PlasticityRule
{
	constexpr static bool can_connect_different_execution_instances = false;

	/**
	 * Timing information for execution of the rule.
	 */
	struct Timer
	{
		/** PPU clock cycles. */
		struct GENPYBIND(inline_base("*")) Value
		    : public halco::common::detail::RantWrapper<Value, uintmax_t, 0xffffffff, 0>
		{
			constexpr explicit Value(uintmax_t const value = 0) : rant_t(value) {}
		};

		Value start;
		Value period;
		size_t num_periods;

		bool operator==(Timer const& other) const SYMBOL_VISIBLE;
		bool operator!=(Timer const& other) const SYMBOL_VISIBLE;

		GENPYBIND(stringstream)
		friend std::ostream& operator<<(std::ostream& os, Timer const& timer) SYMBOL_VISIBLE;

	private:
		friend struct cereal::access;
		template <typename Archive>
		void serialize(Archive& ar, std::uint32_t version);
	};

	/**
	 * Recording information for execution of the rule.
	 * Raw recording of one scratchpad memory region for all timed invocations of the
	 * rule. No automated recording of time is performed.
	 */
	struct RawRecording
	{
		/**
		 * Size (in bytes) of recorded scratchpad memory, which is stored after execution and
		 * provided as output of this vertex.
		 */
		size_t scratchpad_memory_size;

		bool operator==(RawRecording const& other) const SYMBOL_VISIBLE;
		bool operator!=(RawRecording const& other) const SYMBOL_VISIBLE;

		GENPYBIND(stringstream)
		friend std::ostream& operator<<(std::ostream& os, RawRecording const& recording)
		    SYMBOL_VISIBLE;

	private:
		friend struct cereal::access;
		template <typename Archive>
		void serialize(Archive& ar, std::uint32_t version);
	};

	/**
	 * Recording information for execution of the rule.
	 * Recording of exclusive scratchpad memory per rule invocation with
	 * time recording and returned data as time-annotated events.
	 */
	struct TimedRecording
	{
		/**
		 * Observable with a single data entry per synapse.
		 * Used for e.g. weights and correlation measurements
		 */
		struct ObservablePerSynapse
		{
			struct Type
			{
				struct GENPYBIND(inline_base("*")) Int8
				    : plasticity_rule::ObservableDataType<int8_t, Int8>
				{
					constexpr static char GENPYBIND(hidden)
					    on_ppu_type[] = "libnux::vx::VectorRowFracSat8";
				};
				constexpr static Int8 int8{};

				struct GENPYBIND(inline_base("*")) UInt8
				    : plasticity_rule::ObservableDataType<uint8_t, UInt8>
				{
					constexpr static char GENPYBIND(hidden)
					    on_ppu_type[] = "libnux::vx::VectorRowMod8";
				};
				constexpr static UInt8 uint8{};

				struct GENPYBIND(inline_base("*")) Int16
				    : plasticity_rule::ObservableDataType<int16_t, Int16>
				{
					constexpr static char GENPYBIND(hidden)
					    on_ppu_type[] = "libnux::vx::VectorRowFracSat16";
				};
				constexpr static Int16 int16{};

				struct GENPYBIND(inline_base("*")) UInt16
				    : plasticity_rule::ObservableDataType<uint16_t, UInt16>
				{
					constexpr static char GENPYBIND(hidden)
					    on_ppu_type[] = "libnux::vx::VectorRowMod16";
				};
				constexpr static UInt16 uint16{};
			};
			typedef std::variant<Type::Int8, Type::UInt8, Type::Int16, Type::UInt16> TypeVariant;
			TypeVariant type;

			enum class LayoutPerRow
			{
				complete_rows, /** Record complete rows of values. This is fast, but inefficient
				                  memory-wise, since independent of the number of active columns in
				                  the synapse view the complete row is stored. The memory provided
				                  per row is of type libnux::vx::VectorRow{Mod,FracSat}{8,16} for
				                  type {u,}{int_}{8,16}. */
				packed_active_columns /** Record only active columns of synapse view. This is
				                         efficient memory-wise, but slow, since the values are
				                         stored sequentially per synapse. The memory provided per
				                         row is of type std::array<{u,}{int_}{8,16}_t, num_columns>
				                         for type {u,}{int_}{8,16}. */
			} layout_per_row = LayoutPerRow::complete_rows;

			bool operator==(ObservablePerSynapse const& other) const SYMBOL_VISIBLE;
			bool operator!=(ObservablePerSynapse const& other) const SYMBOL_VISIBLE;

			GENPYBIND(stringstream)
			friend std::ostream& operator<<(
			    std::ostream& os, ObservablePerSynapse const& observable) SYMBOL_VISIBLE;

		private:
			friend struct cereal::access;
			template <typename Archive>
			void serialize(Archive& ar, std::uint32_t version);
		};

		/**
		 * Observable with a single data entry per neuron.
		 * Used for e.g. membrane potential measurements
		 */
		struct ObservablePerNeuron
		{
			struct Type
			{
				struct GENPYBIND(inline_base("*")) Int8
				    : plasticity_rule::ObservableDataType<int8_t, Int8>
				{
					constexpr static char GENPYBIND(hidden)
					    on_ppu_type[] = "libnux::vx::VectorRowFracSat8";
				};
				constexpr static Int8 int8{};

				struct GENPYBIND(inline_base("*")) UInt8
				    : plasticity_rule::ObservableDataType<uint8_t, UInt8>
				{
					constexpr static char GENPYBIND(hidden)
					    on_ppu_type[] = "libnux::vx::VectorRowMod8";
				};
				constexpr static UInt8 uint8{};

				struct GENPYBIND(inline_base("*")) Int16
				    : plasticity_rule::ObservableDataType<int16_t, Int16>
				{
					constexpr static char GENPYBIND(hidden)
					    on_ppu_type[] = "libnux::vx::VectorRowFracSat16";
				};
				constexpr static Int16 int16{};

				struct GENPYBIND(inline_base("*")) UInt16
				    : plasticity_rule::ObservableDataType<uint16_t, UInt16>
				{
					constexpr static char GENPYBIND(hidden)
					    on_ppu_type[] = "libnux::vx::VectorRowMod16";
				};
				constexpr static UInt16 uint16{};
			};
			typedef std::variant<Type::Int8, Type::UInt8, Type::Int16, Type::UInt16> TypeVariant;
			TypeVariant type;

			enum class Layout
			{
				complete_row, /** Record complete row of values. This is fast, but inefficient
				                  memory-wise, since independent of the number of active columns in
				                  the neuron view the complete row is stored. The memory provided
				                  per row is of type libnux::vx::VectorRow{Mod,FracSat}{8,16} for
				                  type {u,}{int_}{8,16}. */
				packed_active_columns /** Record only active columns of neuron view. This is
				                         efficient memory-wise, but slow, since the values are
				                         stored sequentially per neuron. The memory provided per
				                         row is of type std::array<{u,}{int_}{8,16}_t, num_columns>
				                         for type {u,}{int_}{8,16}. */
			} layout = Layout::complete_row;

			bool operator==(ObservablePerNeuron const& other) const SYMBOL_VISIBLE;
			bool operator!=(ObservablePerNeuron const& other) const SYMBOL_VISIBLE;

			GENPYBIND(stringstream)
			friend std::ostream& operator<<(std::ostream& os, ObservablePerNeuron const& observable)
			    SYMBOL_VISIBLE;

		private:
			friend struct cereal::access;
			template <typename Archive>
			void serialize(Archive& ar, std::uint32_t version);
		};

		/**
		 * Observable with array of values of configurable size.
		 * Used for e.g. neuron firing rates.
		 */
		struct ObservableArray
		{
			struct Type
			{
				struct GENPYBIND(inline_base("*")) Int8
				    : plasticity_rule::ObservableDataType<int8_t, Int8>
				{
					constexpr static char GENPYBIND(hidden) on_ppu_type[] = "int8_t";
				};
				constexpr static Int8 int8{};

				struct GENPYBIND(inline_base("*")) UInt8
				    : plasticity_rule::ObservableDataType<uint8_t, UInt8>
				{
					constexpr static char GENPYBIND(hidden) on_ppu_type[] = "uint8_t";
				};
				constexpr static UInt8 uint8{};

				struct GENPYBIND(inline_base("*")) Int16
				    : plasticity_rule::ObservableDataType<int16_t, Int16>
				{
					constexpr static char GENPYBIND(hidden) on_ppu_type[] = "int16_t";
				};
				constexpr static Int16 int16{};

				struct GENPYBIND(inline_base("*")) UInt16
				    : plasticity_rule::ObservableDataType<uint16_t, UInt16>
				{
					constexpr static char GENPYBIND(hidden) on_ppu_type[] = "uint16_t";
				};
				constexpr static UInt16 uint16{};
			};
			typedef std::variant<Type::Int8, Type::UInt8, Type::Int16, Type::UInt16> TypeVariant;
			TypeVariant type;

			size_t size;

			bool operator==(ObservableArray const& other) const SYMBOL_VISIBLE;
			bool operator!=(ObservableArray const& other) const SYMBOL_VISIBLE;

			GENPYBIND(stringstream)
			friend std::ostream& operator<<(std::ostream& os, ObservableArray const& observable)
			    SYMBOL_VISIBLE;

		private:
			friend struct cereal::access;
			template <typename Archive>
			void serialize(Archive& ar, std::uint32_t version);
		};

		/**
		 * Observable type specification.
		 */
		typedef std::variant<ObservablePerSynapse, ObservablePerNeuron, ObservableArray> Observable;

		/**
		 * Map of named observables with type information.
		 * The plasticity rule kernel is given memory to record the observables generated with the
		 * same names as used in this map.
		 */
		std::map<std::string, Observable> observables;

		bool operator==(TimedRecording const& other) const SYMBOL_VISIBLE;
		bool operator!=(TimedRecording const& other) const SYMBOL_VISIBLE;

		GENPYBIND(stringstream)
		friend std::ostream& operator<<(std::ostream& os, TimedRecording const& recording)
		    SYMBOL_VISIBLE;

	private:
		friend struct cereal::access;
		template <typename Archive>
		void serialize(Archive& ar, std::uint32_t version);
	};

	/**
	 * Recording information for execution of the rule.
	 */
	typedef std::variant<RawRecording, TimedRecording> Recording;

	/**
	 * Recording data corresponding to a raw recording.
	 */
	struct RawRecordingData
	{
		/**
		 * Data with outer dimension being batch entries and inner dimension being the data per
		 * batch entry.
		 */
		std::vector<std::vector<Int8>> data;
	};

	/**
	 * Extracted recorded data of observables corresponding to timed recording.
	 */
	struct TimedRecordingData
	{
		typedef std::variant<
		    std::vector<TimedDataSequence<std::vector<int8_t>>>,
		    std::vector<TimedDataSequence<std::vector<uint8_t>>>,
		    std::vector<TimedDataSequence<std::vector<int16_t>>>,
		    std::vector<TimedDataSequence<std::vector<uint16_t>>>>
		    Entry;

		std::map<std::string, std::vector<Entry>> data_per_synapse;
		std::map<std::string, std::vector<Entry>> data_per_neuron;
		std::map<std::string, Entry> data_array;
	};

	typedef std::variant<RawRecordingData, TimedRecordingData> RecordingData;

	/**
	 * Shape of a single synapse view to be altered.
	 */
	struct SynapseViewShape
	{
		/** Number of rows. */
		size_t num_rows;
		/**
		 * Location of columns.
		 * This information is needed for extraction of timed recording observables.
		 */
		std::vector<halco::hicann_dls::vx::v3::SynapseOnSynapseRow> columns;
		/**
		 * Hemisphere of synapses.
		 * This information is required for extraction of timed recording observables.
		 */
		halco::hicann_dls::vx::v3::HemisphereOnDLS hemisphere;

		bool operator==(SynapseViewShape const& other) const SYMBOL_VISIBLE;
		bool operator!=(SynapseViewShape const& other) const SYMBOL_VISIBLE;

		GENPYBIND(stringstream)
		friend std::ostream& operator<<(std::ostream& os, SynapseViewShape const& recording)
		    SYMBOL_VISIBLE;

	private:
		friend struct cereal::access;
		template <typename Archive>
		void serialize(Archive& ar, std::uint32_t version);
	};

	/**
	 * Shape of a single neuron view to be altered.
	 */
	struct NeuronViewShape
	{
		/**
		 * Location of columns.
		 * This information is needed for extraction of timed recording observables and measurement
		 * settings.
		 */
		NeuronView::Columns columns;
		/**
		 * Row of neurons.
		 * This information is required for extraction of timed recording observables and
		 * measurement settings.
		 */
		NeuronView::Row row;

		typedef lola::vx::v3::AtomicNeuron::Readout::Source NeuronReadoutSource;
		/**
		 * Readout source specification per neuron used for static configuration such that the
		 * plasticity rule can read the specified signal.
		 */
		std::vector<std::optional<NeuronReadoutSource>> neuron_readout_sources;

		bool operator==(NeuronViewShape const& other) const SYMBOL_VISIBLE;
		bool operator!=(NeuronViewShape const& other) const SYMBOL_VISIBLE;

		GENPYBIND(stringstream)
		friend std::ostream& operator<<(std::ostream& os, NeuronViewShape const& recording)
		    SYMBOL_VISIBLE;

	private:
		friend struct cereal::access;
		template <typename Archive>
		void serialize(Archive& ar, std::uint32_t version);
	};

	PlasticityRule() = default;

	/**
	 * Construct PlasticityRule with specified kernel, timer and synapse information.
	 * @param kernel Kernel to apply
	 * @param timer Timer to use
	 * @param synapse_view_shapes Shapes of synapse views to alter
	 * @param neuron_view_shapes Shapes of neuron views to alter
	 * @param recording Optional recording providing memory for the plasticity rule to store
	 * information during execution.
	 */
	PlasticityRule(
	    std::string kernel,
	    Timer const& timer,
	    std::vector<SynapseViewShape> const& synapse_view_shapes,
	    std::vector<NeuronViewShape> const& neuron_view_shapes,
	    std::optional<Recording> const& recording) SYMBOL_VISIBLE;

	/**
	 * Size (in bytes) of recorded scratchpad memory, which is stored after execution and
	 * provided as output of this vertex.
	 * When the recording is TimedRecording, the shapes of the synapse views are used to
	 * calculate the space requirement for the observables per synapse.
	 */
	size_t get_recorded_scratchpad_memory_size() const SYMBOL_VISIBLE;

	/**
	 * Alignment (in bytes) of recorded scratchpad memory.
	 */
	size_t get_recorded_scratchpad_memory_alignment() const SYMBOL_VISIBLE;

	/**
	 * Get C++ definition of recorded memory structure.
	 * This structure is instantiated and made available to the plasticity rule kernel.
	 */
	std::string get_recorded_memory_definition() const SYMBOL_VISIBLE;

	/**
	 * Get interval in memory layout of data within recording.
	 */
	std::pair<size_t, size_t> get_recorded_memory_data_interval() const SYMBOL_VISIBLE;

	/**
	 * Get interval in memory layout of data within recorded events.
	 * Data intervals are given per observable.
	 * @throws std::runtime_error On no or raw recording type
	 */
	std::map<std::string, std::pair<size_t, size_t>> get_recorded_memory_timed_data_intervals()
	    const SYMBOL_VISIBLE;

	/**
	 * Extract data corresponding to performed recording.
	 * For RawRecording return the raw data, for TimedRecording extract observables of timed
	 * recording from raw data. This method is to be used after successful execution of a graph
	 * incorporating this vertex instance.
	 * @throws std::runtime_error On data not matching expectation
	 * @param data Raw data to extract recording from
	 */
	RecordingData extract_recording_data(
	    std::vector<TimedDataSequence<std::vector<Int8>>> const& data) const SYMBOL_VISIBLE;

	std::string const& get_kernel() const SYMBOL_VISIBLE;
	Timer const& get_timer() const SYMBOL_VISIBLE;
	std::vector<SynapseViewShape> const& get_synapse_view_shapes() const SYMBOL_VISIBLE;
	std::vector<NeuronViewShape> const& get_neuron_view_shapes() const SYMBOL_VISIBLE;
	std::optional<Recording> get_recording() const SYMBOL_VISIBLE;

	constexpr static bool variadic_input = false;
	std::vector<Port> inputs() const SYMBOL_VISIBLE;

	Port output() const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, PlasticityRule const& config) SYMBOL_VISIBLE;

	bool supports_input_from(
	    SynapseArrayView const& input,
	    std::optional<PortRestriction> const& restriction) const SYMBOL_VISIBLE;

	bool supports_input_from(
	    NeuronView const& input,
	    std::optional<PortRestriction> const& restriction) const SYMBOL_VISIBLE;

	bool operator==(PlasticityRule const& other) const SYMBOL_VISIBLE;
	bool operator!=(PlasticityRule const& other) const SYMBOL_VISIBLE;

private:
	std::string m_kernel;
	Timer m_timer;
	std::vector<SynapseViewShape> m_synapse_view_shapes;
	std::vector<NeuronViewShape> m_neuron_view_shapes;
	std::optional<Recording> m_recording;

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

std::ostream& operator<<(
    std::ostream& os,
    PlasticityRule::TimedRecording::ObservablePerSynapse::LayoutPerRow const& layout)
    SYMBOL_VISIBLE;

std::ostream& operator<<(
    std::ostream& os,
    PlasticityRule::TimedRecording::ObservablePerNeuron::Layout const& layout) SYMBOL_VISIBLE;

} // vertex

} // grenade::vx
