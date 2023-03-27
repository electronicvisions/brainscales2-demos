#pragma once
#include "hate/visibility.h"
#include "stadls/vx/run_time_info.h"

namespace stadls::vx {
struct PlaybackProgram;

namespace v3 {
using PlaybackProgram = stadls::vx::PlaybackProgram;
} // namespace v3

} // namespace stadls::vx

namespace grenade::vx::backend {

struct Connection;

/**
 * Execute given playback program using the given connection.
 * @param connection Connection to run on
 * @param program Program to execute
 * @return Run time information of execution
 */
stadls::vx::RunTimeInfo run(Connection& connection, stadls::vx::v3::PlaybackProgram& program)
    SYMBOL_VISIBLE;

/**
 * Execute given playback program using the given connection.
 * @param connection Connection to run on
 * @param program Program to execute
 * @return Run time information of execution
 */
stadls::vx::RunTimeInfo run(Connection& connection, stadls::vx::v3::PlaybackProgram&& program)
    SYMBOL_VISIBLE;

} // namespace grenade::vx::backend
