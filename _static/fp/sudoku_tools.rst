.. code:: ipython3

    from functools import partial
    from collections import OrderedDict
    from ipywidgets import interactive, IntSlider, FloatSlider, Layout,\
        VBox, Box, HTML


    layout = {"width": "calc(100% - 10px)"}
    style = {"description_width": "120px"}
    IntSlider = partial(IntSlider, continuous_update=False)
    controls_parameter = OrderedDict([
        ("num_clues", ("Number Hints", (0, 16), 8, "config")),
        ("random_seed", ("Random Seed", (0, 65636), 1234, "config")),
        ("runtime", ("Runtime", (.1, 10), .5, "config"))
    ])
    headers = {
        "config": "Configuration",
    }
    
    
    def build_gui(callback,
                  controls,
                  config=None,
                  defaults=None,
                  copy_configuration=False):
        """
        Build a slider-based GUI for an experiment callback function.
        The sliders are grouped by the specific circuit they affect and the
        callback's result (e.g. graph) is displayed above the sliders.
        """
    
        print(config)
    
        if config is None:
            config = {}
        if defaults is None:
            defaults = {}
    
        # instantiate sliders according to list of parameters provided by the user
        sliders = OrderedDict()
        for con in controls:
            spec = controls_parameter[con]
            default = defaults[con] if con in defaults else spec[2]
            if copy_configuration and con in last_configuration:
                default = last_configuration[con]
    
            if con != "runtime":
                sliders[con] = IntSlider(min=spec[1][0],
                                         max=spec[1][1],
                                         step=1,
                                         value=default,
                                         description=spec[0],
                                         layout=layout,
                                         style=style)
            else:
                sliders[con] = FloatSlider(min=spec[1][0],
                                           max=spec[1][1],
                                           step=.1,
                                           value=default,
                                           description=spec[0],
                                           layout=layout,
                                           style=style)
    
        widget = interactive(callback, **sliders, **config)
    
        # group sliders according to their sections
        sections = OrderedDict()
        for con, slider in sliders.items():
            header = controls_parameter[con][3]
            if header not in sections:
                sections[header] = OrderedDict()
            sections[header][con] = slider
    
        # build UI according to hierarchical structure from above
        u_i = []
        for header, children in sections.items():
            u_i.append([])
    
            u_i[-1].append(HTML(f"<h3>{headers[header]}</h3>"))
    
            for slider in children.values():
                u_i[-1].append(slider)
    
        output = widget.children[-1]
    
        # define custom layout following the responsive web design paradigm
        slider_box = Box(tuple(VBox(tuple(s)) for s in u_i), layout=Layout(
            display='grid',
            grid_template_columns='repeat(auto-fit, minmax(400px, 1fr))',
            width='100%'
        ))
    
        display(VBox([slider_box, output]))
    
        widget.update()
