from collections.abc import Callable
from dataclasses import fields, is_dataclass, make_dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from dacite import Config as DaciteConfig
from dacite import from_dict


def string_to_class_name(s: str) -> str:
    return ''.join(word.capitalize() for word in s.split('_'))


if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    C = TypeVar("C", bound=DataclassInstance)
    MetaKey = DataclassInstance
else:
    C = TypeVar("C")
    MetaKey = Any


# meta registry keeps track of all registries
_meta_registry: dict[MetaKey, "src.registry.Registry"] = {}

def get_type_hooks(
    dacite_config: DaciteConfig | None = None
) -> dict[MetaKey, Callable[[dict[str, Any]], MetaKey]]:
    global _meta_registry
    return {n: partial(r.build_config_from_dict, dacite_config=dacite_config) for n, r in _meta_registry.items()}


T = TypeVar("T")


class Registry(Generic[T, C]):
    def __init__(
        self, 
        ref_cls: type[T],
        ref_config_cls: type[C],
        key_str: str = "name"
    ) -> None:
        if not is_dataclass(ref_config_cls):
            raise ValueError(f"Reference config class {self.name} must be a dataclass")
        self._ref_cls = ref_cls
        self._ref_config_cls = ref_config_cls
        
        # Register registry in meta_registry
        global _meta_registry
        if self.ref_config_cls in _meta_registry:
            raise KeyError(f"There is already a registry for the config class {self.name}")
        _meta_registry[self.ref_config_cls] = self

        self._key_str = key_str
        self._name_to_config: dict[str, type[C]] = {}
        self._config_to_class: dict[type[C], type[T]] = {}

    @property
    def name(self) -> str:
        return self._ref_config_cls.__name__

    @property
    def ref_cls(self) -> type:
        return self._ref_cls

    @property
    def ref_config_cls(self) -> type:
        return self._ref_config_cls

    @property
    def key_str(self) -> str:
        return self._key_str

    def _register(
        self,
        cls: type[T],
        name: str | None = None,
        config: type[C] | None = None,
        force: bool = False
    ) -> type[T]:
        if not issubclass(cls, self._ref_cls):
            raise ValueError(
                f"Registered class {cls.__name__} must be a subclass of {self._ref_cls.__name__}"
            )
        if name is None:
            name = cls.__name__
        if not force and name in self._name_to_config:
            raise KeyError(
                f"Registry for config {self.name} already contains a config \
                    ({self._name_to_config[name].__name__}) with name {name}"
            )
        if config is None:
            # Create new subclass for bijective mapping between configs and classes
            config = make_dataclass(
                string_to_class_name(name) + "Config", [], bases=(self._ref_config_cls,)
            )
        else:
            if not issubclass(config, self._ref_config_cls):
                raise ValueError(
                    f"Config class {config.__name__} must be a subclass of {self._ref_config_cls.__name__}"
                )
            if self.key_str in config.__dataclass_fields__:
                raise ValueError(
                    f"Config class {config.__name__} contains special key {self.key_str} used for the registry"
                )
            if not force and config in self._config_to_class:
                raise KeyError(
                    f"Registry for config {self.name} already contains a class \
                        ({self._config_to_class[config].__name__}) with config {config.__name__}"
                )
            
        self._name_to_config[name] = config
        self._config_to_class[config] = cls
        return cls

    def register(
        self,
        name: str | None = None,
        config: type[C] | None = None,
        item: type[T] | None = None,
        force: bool = False
    ) -> Callable[[type[T]], type[T]] | type[T]:
        # use it as a normal method: r.register(item=MyItemClass)
        if item is not None:
            self._register(cls=item, name=name, config=config, force=force)
            return item
        
        # use it as a decorator: @r.register()
        return partial(self._register, name=name, config=config, force=force)

    def build_config(
        self,
        name: str,
        **kwargs: Any
    ) -> C:
        return self._name_to_config[name](**kwargs)

    def build_config_from_dict(
        self, 
        d: dict[str, Any],
        dacite_config: DaciteConfig | None = None
    ) -> C:
        """NOTE will do strict matching as default if dacite_config is None"""
        if self.key_str not in d:
            raise ValueError(f"Dictionary {d} does not contain special key {self.key_str}")
        
        registry_key = d[self.key_str]
        if registry_key not in self._name_to_config:
            raise ValueError(
                f"Registry {self.name} does not contain a config with name {registry_key},"
                " possible values are: "
                f"{', '.join(self._name_to_config.keys())}"
            )
        
        config_cls = self._name_to_config[registry_key]
        if dacite_config is None:
            dacite_config = DaciteConfig(type_hooks=get_type_hooks(), strict=True)
        else:
            dacite_config = DaciteConfig(
                type_hooks=dacite_config.type_hooks | get_type_hooks(dacite_config),
                **{f.name: getattr(dacite_config, f.name) for f in fields(dacite_config) if f.name != "type_hooks"}
            )
        return from_dict(config_cls, {k: v for k, v in d.items() if k != self.key_str}, config=dacite_config)

    def get(
        self,
        config: C
    ) -> type[T]:
        return self._config_to_class[config.__class__]

    def build(
        self,
        config: C,
        *args: Any,
        **kwargs: Any
    ) -> T:
        return self.get(config)(config, *args, **kwargs)
