import numpy as np

from dlens_vx_v3 import halco, hal, lola, sta, hxcomm


# Function to apply an external voltage and measure it on the MADC
# You don't need to make changes in this function.
def measure_voltage(
    connection: hxcomm.ConnectionHandle,
    voltage: float
) -> np.ndarray:
    """
    Apply the given voltage to the MADC and return a few acquired samples.

    The first 120 received samples are discarded as they may not be valid
    already, cf. issue 4008.

    :param connection: Connection to the chip.
    :param voltage: Voltage in Volt, applied externally, via the DAC
                    on the xboard.

    :returns: Array of acquired MADC samples.
    """

    # connect DAC to chip, set voltage on DAC
    # pylint: disable=no-member
    init = sta.ExperimentInit()
    init.asic_adapter_board.shift_register\
        .select_analog_readout_mux_1_input = \
        hal.ShiftRegister.AnalogReadoutMux1Input.readout_chain_0
    init.asic_adapter_board.shift_register\
        .select_analog_readout_mux_2_input = \
        hal.ShiftRegister.AnalogReadoutMux2Input.v_reset
    # pylint: enable=no-member
    init.asic_adapter_board.dac_channel_block.set_voltage(  # pylint: disable=no-member
        halco.DACChannelOnBoard.mux_dac_25,
        voltage)
    builder, _ = init.generate()

    # connect MADC to readout pads
    readout_config = lola.ReadoutChain()
    readout_config.pad_mux[  # pylint: disable=unsubscriptable-object
        halco.PadMultiplexerConfigOnDLS()
    ].debug_to_pad = True
    readout_config.input_mux[  # pylint: disable=unsubscriptable-object
        halco.SourceMultiplexerOnReadoutSourceSelection()].debug_plus = True
    readout_config.madc.number_of_samples = 420  # record a few samples
    builder.write(halco.ReadoutChainOnDLS(), readout_config)

    # wake up MADC
    madc_control = hal.MADCControl()
    madc_control.enable_power_down_after_sampling = False
    madc_control.start_recording = False
    madc_control.wake_up = True
    madc_control.enable_pre_amplifier = True
    builder.write(halco.MADCControlOnDLS(), madc_control)

    # initial wait (includes wait for CapMem), systime sync
    builder.write(halco.TimerOnDLS(), hal.Timer())
    builder.block_until(halco.TimerOnDLS(), hal.Timer.Value(
        100000 * int(hal.Timer.Value.fpga_clock_cycles_per_us)))
    builder.write(halco.SystimeSyncOnFPGA(), hal.SystimeSync())

    # enable recording of samples
    config = hal.EventRecordingConfig()
    config.enable_event_recording = True
    builder.write(halco.EventRecordingConfigOnFPGA(), config)

    # trigger MADC sampling
    madc_control.wake_up = True
    madc_control.start_recording = True
    madc_control.enable_power_down_after_sampling = True
    builder.write(halco.MADCControlOnDLS(), madc_control)

    # wait for samples
    builder.write(halco.TimerOnDLS(), hal.Timer())
    builder.block_until(halco.TimerOnDLS(), hal.Timer.Value(
        1000 * int(hal.Timer.Value.fpga_clock_cycles_per_us)))

    # disable recording of samples
    config = hal.EventRecordingConfig()
    config.enable_event_recording = False
    builder.write(halco.EventRecordingConfigOnFPGA(), config)

    # run program
    program = builder.done()
    sta.run(connection, program)

    return program.madc_samples.to_numpy()[120:]
