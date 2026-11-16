"""External data that cannot be redistributed: read from the environment, never guessed."""
import os


def external(name, env, hint):
    """Path of an external file or directory; stop with a message naming the variable to set."""
    value = os.environ.get(env)
    if not value:
        raise RuntimeError("%s is not set: %s" % (env, hint))
    return os.path.join(value, name) if name else value
