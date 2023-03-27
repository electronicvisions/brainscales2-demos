#pragma once
#include "hate/visibility.h"
#include <exception>
#include <string>

namespace grenade::vx::network {

/**
 * Exception describing an invalid network graph.
 */
class SYMBOL_VISIBLE InvalidNetworkGraph : public virtual std::exception
{
public:
	/**
	 * Construct from cause message.
	 * @param message Exception cause
	 */
	explicit InvalidNetworkGraph(std::string const& message);

	/**
	 * Get exception cause.
	 * @return String describing cause of exception
	 */
	virtual const char* what() const noexcept override;

private:
	std::string const m_message;
};


/**
 * Exception describing an unsuccessful routing.
 */
class SYMBOL_VISIBLE UnsuccessfulRouting : public virtual std::exception
{
public:
	/**
	 * Construct from cause message.
	 * @param message Exception cause
	 */
	explicit UnsuccessfulRouting(std::string const& message);

	/**
	 * Get exception cause.
	 * @return String describing cause of exception
	 */
	virtual const char* what() const noexcept override;

private:
	std::string const m_message;
};

} // namespace grenade::vx::network
