#include "grenade/vx/genpybind.h"

GENPYBIND_TAG_GRENADE_VX
GENPYBIND_MANUAL({
	parent.attr("__variant__") = "pybind11";
	parent->py::module::import("pyhalco_hicann_dls_vx_v3");
	parent->py::module::import("pyhaldls_vx_v3");
	parent->py::module::import("pylola_vx_v3");
})

#include "grenade/vx/grenade.h"
