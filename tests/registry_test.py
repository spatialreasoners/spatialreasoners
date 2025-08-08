from dataclasses import dataclass, fields, make_dataclass

from dacite import Config, from_dict
import pytest

from src.registry import Registry, get_type_hooks, _meta_registry


@dataclass
class BaseConfig:
    base_nondefault: int

class BaseClass:
    pass


@dataclass
class MyConfig1(BaseConfig):
    my_nondefault1: str
    my_default1: str = "hello"


class MyClass1(BaseClass):
    pass


@dataclass
class MyConfig2(BaseConfig):
    my_nondefault2: str
    my_default2: str = "world"


class MyClass2(BaseClass):
    pass


@pytest.fixture(params=["name", "key"])
def sample_registry(request: pytest.FixtureRequest):
    _meta_registry.clear()
    return Registry(BaseClass, BaseConfig, key_str=request.param)


def test_key_string_collision(sample_registry: Registry):
    key_collision_config = make_dataclass(
        "KeyCollisionConfig", 
        fields=[(sample_registry.key_str, str)], 
        bases=(BaseConfig,)
    )
    class KeyCollisionClass(BaseClass):
        pass
    with pytest.raises(ValueError):
        sample_registry.register("my_class", key_collision_config, KeyCollisionClass)


def test_key_collision(sample_registry: Registry):
    sample_registry.register("my_class", MyConfig1, MyClass1)
    with pytest.raises(KeyError):
        sample_registry.register("my_class", MyConfig2, MyClass2)


def test_config_is_not_subclass(sample_registry: Registry):
    @dataclass
    class NotSubclassConfig:
        pass
    class MyClass(BaseClass):
        pass
    with pytest.raises(ValueError):
        sample_registry.register("my_class", NotSubclassConfig, MyClass)


def test_is_not_subclass(sample_registry: Registry):
    @dataclass
    class MyConfig(BaseConfig):
        pass
    class NotSubclass:
        pass
    with pytest.raises(ValueError):
        sample_registry.register("my_class", MyConfig, NotSubclass)
 

@pytest.mark.parametrize(
    "name1, name2",
    [("my_class1", "my_class2"), ("your_class1", "your_class2")]
)
def test_from_dict(sample_registry: Registry, name1: str, name2: str):
    sample_registry.register(name1, MyConfig1, MyClass1)
    sample_registry.register(name2, MyConfig2, MyClass2)
    @dataclass
    class WrapperConfig:
        a: BaseConfig
        b: BaseConfig
    
    a_dict = {
        sample_registry.key_str: name1,
        "my_nondefault1": "a",
        "base_nondefault": 2
    }
    b_dict = {
        sample_registry.key_str: name2,
        "my_nondefault2": "b",
        "base_nondefault": 3
    }
    d = {
        "a": a_dict,
        "b": b_dict
    }
    config = from_dict(WrapperConfig, d, Config(type_hooks=get_type_hooks()))

    def get_default_fields(dataclass_type):
        default_fields = {
            field.name: field.default
            for field in fields(dataclass_type)
            if field.default is not field.default_factory
        }
        return default_fields

    assert isinstance(config.a, MyConfig1)
    assert config.a.__dict__ == get_default_fields(MyConfig1) | {k: v for k, v in a_dict.items() if k != sample_registry.key_str}
    assert isinstance(config.b, MyConfig2)
    assert config.b.__dict__ == get_default_fields(MyConfig2) | {k: v for k, v in b_dict.items() if k != sample_registry.key_str}
