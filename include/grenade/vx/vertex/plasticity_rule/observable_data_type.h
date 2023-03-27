#pragma once
#include "grenade/vx/genpybind.h"
#include "hate/type_index.h"
#include <cctype>
#include <ostream>

namespace cereal {
struct access;
} // namespace cereal

namespace grenade::vx::vertex::plasticity_rule {

template <typename ElementT, typename Derived>
struct ObservableDataType
{
	typedef ElementT ElementType;

	friend constexpr bool operator==(Derived const&, Derived const&)
	{
		return true;
	}

	friend constexpr bool operator!=(Derived const&, Derived const&)
	{
		return false;
	}

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, Derived const&)
	{
		auto name = hate::name<Derived>();
		std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
			return std::tolower(c);
		});
		return os << name;
	}

private:
	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive&)
	{}
};

} // grenade::vx::vertex::plasticity_rule
