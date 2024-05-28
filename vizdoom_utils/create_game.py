import vizdoom as vzd

def create_game(config_path: str, color: bool|str=False, depth: bool=False, label: bool=False, 
                automap: bool=False, res: tuple[int, int]=(256, 144), visibility: bool=False, *args, **kwargs) -> vzd.vizdoom.DoomGame:
    """Creates a vizdoom game instance with configurable options

    Args:
        config_path (str): scenario configuration path
        color (bool|str, optional): color (RGB24) or grayscale (GRAY8). Defaults to False.
        depth (bool, optional): depth buffer availability. Defaults to False.
        label (bool, optional): labels buffer availability. Defaults to False.
        automap (bool, optional): automap buffer availability. Defaults to False.
        res (tuple[int, int], optional): internal rendering resolution. Defaults to (256, 144).
        visibility (bool, optional): whether to display game window. Defaults to False.
        asyn (NoneType, optional): keyword argument, if present would set game to asynchronus mode.
        spec (NoneType, optional): keyword argument, if present would set game to spectator mode.
    Returns:
        vzd.vizdoom.DoomGame: _description_
    """    
    
    # create an instance of doom game
    print("Initializing Doom... ", end='')
    game = vzd.DoomGame()
    
    # remove named arguments set to boolean False or anything representing that
    for key, value in kwargs.items():
        if not value:
            del kwargs[key]
    
    # load scenario's default configuration
    game.load_config(config_path)
    
    # set game to sync/async player mode or async spectator mode
    game.set_mode(vzd.Mode.PLAYER) if "asyn" not in kwargs else \
        (game.set_mode(vzd.Mode.ASYNC_PLAYER) if "spec" not in kwargs else \
            game.set_mode(vzd.Mode.ASYNC_SPECTATOR))
    
    # set rendering options
    exec(f"game.set_screen_resolution(vzd.ScreenResolution.RES_{res[0]:d}X{res[1]:d})")
    if isinstance(color, bool):
        exec(f"game.set_screen_format(vzd.ScreenFormat.{'RGB24' if color else 'GRAY8'})")
    else:
        exec(f"game.set_screen_format(vzd.ScreenFormat.{color})")
    game.set_window_visible(visibility)
    
    # set "cheating" buffer access
    game.set_depth_buffer_enabled(depth)
    game.set_labels_buffer_enabled(label)
    game.set_automap_buffer_enabled(automap)
    if "automap_config" in kwargs:
        config = kwargs["automap_config"]
        game.set_automap_mode(config["mode"])
        game.set_automap_rotate(config["rotate"])
        game.set_automap_render_textures(config["texture"])
    
    # finished initializing the game
    game.init()
    print("done")
    return game