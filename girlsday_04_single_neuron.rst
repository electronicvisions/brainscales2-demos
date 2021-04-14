Experimente mit einer einzelnen Nervenzelle
===========================================

1. Einfluss der Zellparameter
-----------------------------

Zuerst wollen wir uns genauer mit der Dynamik einer einzelnen
Nervenzelle auseinandersetzen. Dazu verändern wir ihre Zellparameter und
schauen uns das resultierende Membranpotential an.

.. code:: ipython3

    import pynn_brainscales.brainscales2 as pynn
    import matplotlib.pyplot as plt
    
    
    ##################################################################################
    # Diese Parameter sollen verändert werden.
    neuron_parameters = {                          #                         Bereich
        "leak_v_leak": 400,                        # Ruhepotential          (300-1000)
        "leak_i_bias": 200,                        # Ruhestrom              (0-1022)
        "threshold_v_threshold": 400,              # Schwellenspannung      (0-600)
        "threshold_enable": True,                  # Vergleichsaktivierung
        "refractory_period_refractory_time": 100,  # Refraktärzeit          (0-255)
        "reset_v_reset": 300,                      # Umkehrspannung         (300-1000)
        "reset_i_bias": 1000,                      # Umkehrstrom            (0-1022)
        "membrane_capacitance_capacitance": 63     # Membrankapazität       (0-63)
        }
    ##################################################################################
    
    atomic, inject = pynn.helper.filtered_cocos_from_nightly()
    config_injection = pynn.InjectedConfiguration(
        post_non_realtime=inject)
    pynn.setup(injected_config=config_injection)
    
    pop = pynn.Population(1, pynn.cells.HXNeuron(atomic, **neuron_parameters))
    pop.record(["spikes", "v"])
    
    # Die Laufzeit kann auch angepasst werden.
    pynn.run(0.1)
    
    spiketrain = pop.get_data("spikes").segments[0].spiketrains[0]
    print(f"Das Neuron hat {len(spiketrain)} mal gefeuert.")
    print(f"Die Zeitpunkte der Spikes waren: {spiketrain}")
    
    mem_v = pop.get_data("v").segments[0]
    times, membrane = zip(*mem_v.filter(name="v")[0])
    
    plt.figure()
    plt.plot(times, membrane)
    plt.xlabel("Zeit [ms]")
    plt.ylabel("Membranpotential [LSB]")
    plt.show()
    
    pynn.end()

a) Was ist zu sehen? Wieso ist das so? Was erwartet ihr zu sehen?
   Beachtet dabei, dass auf allen Signalen auch ein Rauschen vorliegen
   kann. Dieses kann Veränderungen im Bereich von etwa 20 Hardware
   Einheiten bewirken, ohne dass diese jedoch etwas bedeuten.
b) Welche Spannung ist dargestellt? Überlegt euch, welche Werte das
   Membranpotential beeinflussen und überprüft eure Vermutungen.
   Dazu ist es hilfreich, sich das Aktionspotential nochmal
   anzuschauen.

.. raw:: html

    <img src="_static/girlsday_actionpotential.svg" width="500"/>

c) Nun soll das Ruhepotential auf seinen Maximalwert gesetzt werden, der
   über der Schwellenspannung liegt. Überlegt euch vorher, was für einen
   Verlauf ihr dafür erwartet.
d) Beobachtet in diesem Modus die Auswirkungen, welche die einzelnen
   Parameter auf die Dynamik haben.

2. Stimulierung einer Nervenzelle
---------------------------------

Nun wird unsere Nervenzelle mit anderen Neuronen verbunden, deren
Feuerverhalten wir einstellen können. Wir wollen beobachten, wie sich
Spikes der Senderpopulation auf die empfangende Nervenzelle auswirken.
Neben den Spikezeiten der Sender Population, kann die Anzahl der
Neuronen, die sie beinhaltet variiert werden. Des Weiteren kann das
synaptische Gewicht, also die Stärke der Verbindung, eingestellt werden.
Eine wichtige Rolle spielt auch die Art, wie der Stimulus interpretiert
wird, ob exzitatorisch oder inhibitorisch.

.. code:: ipython3

    import pynn_brainscales.brainscales2 as pynn
    import matplotlib.pyplot as plt
    
    
    atomic, inject = pynn.helper.filtered_cocos_from_nightly()
    config_injection = pynn.InjectedConfiguration(
        post_non_realtime=inject)
    pynn.setup(injected_config=config_injection)
    
    # Nun muss das Ruhepotential wieder unter die Schwellenspannung gesetzt werden.
    neuron_parameters = {                          #                         Bereich
        "leak_v_leak": 400,                        # Ruhepotential          (300-1000)
        "leak_i_bias": 200,                        # Ruhestrom              (0-1022)
        "threshold_v_threshold": 400,              # Schwellenspannung      (0-600)
        "threshold_enable": True,                  # Vergleichsaktivierung
        "refractory_period_refractory_time": 100,  # Refraktärzeit          (0-255)
        "reset_v_reset": 300,                      # Umkehrspannung         (300-1000)
        "reset_i_bias": 1000,                      # Umkehrstrom            (0-1022)
        "membrane_capacitance_capacitance": 63     # Membrankapazität       (0-63)
        }
    
    # Das ist das Neuron, das wir beobachten werden.
    pop = pynn.Population(1, pynn.cells.HXNeuron(atomic, **neuron_parameters))
    pop.record(["spikes", "v"])
    
    # Das ist die Sender Population, die zu vorgegebenen Spikezeiten einen Stimulus generiert.
    # Die Spikezeiten und die Populationsgröße sollen verändert werden.
    spike_times = [0.01, 0.03, 0.05, 0.07, 0.09]
    src_size = 5
    src = pynn.Population(src_size, pynn.cells.SpikeSourceArray(spike_times=spike_times))
    
    # Das synaptische Gewicht kann zwischen 0 und 63 variiert werden.
    synapse_weight = 32
    synapse = pynn.synapses.StaticSynapse(weight=synapse_weight)
    
    # Die Sender Population 'src' wird mit dem Neuron in 'pop' verbunden.
    pynn.Projection(src, pop, pynn.AllToAllConnector(), 
                    synapse_type=synapse, receptor_type="excitatory")
    
    # Die Laufzeit kann auch wieder verändert werden.
    pynn.run(0.1)
    
    # Das Ergebnis wird ausgegeben.
    spiketrain = pop.get_data("spikes").segments[0].spiketrains[0]
    print(f"Das Neuron hat {len(spiketrain)} mal gefeuert.")
    print(f"Die Zeitpunkte der Spikes waren: {spiketrain}")
    
    mem_v = pop.get_data("v").segments[0]
    times, membrane = zip(*mem_v.filter(name="v")[0])
    
    plt.figure()
    plt.plot(times, membrane)
    plt.xlabel("Zeit [ms]")
    plt.ylabel("Membranpotential [LSB]")
    plt.show()
    
    pynn.end()

a) Ist zu den eingestellten Spikezeiten der Senderpopulation eine
   Reaktion im Membranpotential der beobachteten Nervenzelle zu sehen?
   Feuert es selbst auch schon?
b) Was geschieht, wenn man in der Projektion den ``receptor_type`` auf
   ``"inhibitory"`` stellt?
c) Nun wollen wir das Neuron zum Feuern bringen. Dazu wird der
   ‘receptor_type’ wieder auf ``"excitatory"`` gestellt. Ein erster
   Ansatz um das Neuron zum Feuern zu bringen ist die Anzahl der
   Partner, von denen es Spikes erhält, zu erhöhen. Ab welcher Größe der
   Sender Population treten die ersten Spikes auf?
d) Eine weitere Möglichkeit ist das synaptische Gewicht anzupassen.
   Stellt dazu wieder eine kleinere Populationsgröße ein und testet, ob
   ihr durch Erhöhung des synaptischen Gewichts das Neuron zum Feuern
   bringen könnt.
e) Als letztes soll noch untersucht werden, was für Auswirkungen es hat,
   wenn man die Spikezeiten der Sender Population näher zusammen
   schiebt. Probiert hier auch unterschiedliche Abstände zwischen den
   einzelnen Spikes aus. Gegebenfalls müsst ihr hier auch nochmal die
   Neuronparameter anpassen, um einen schönen Verlauf der
   Membranspannung zu bekommen.
