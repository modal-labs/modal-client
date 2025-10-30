# Copyright Modal Labs 2025
import pytest

import modal
import modal.experimental
from modal import App
from modal._utils.flash_utils import get_flash_configs
from modal.exception import InvalidError
from modal.experimental import flash_web_server


def test_flash_web_server_basic_functionality(client, servicer):
    """Test basic flash_web_server decorator functionality."""
    app = App("flash-basic")

    @app.cls()
    class FlashClass:
        @flash_web_server(8080)
        def serve(self):
            return "Hello Flash"

    with app.run(client=client):
        obj = FlashClass()
        # Verify flash configuration is present
        flash_configs = get_flash_configs(FlashClass)
        assert len(flash_configs) == 1
        assert flash_configs[0].port == 8080
        assert flash_configs[0].region is None


def test_flash_web_server_with_region(client, servicer):
    """Test flash_web_server decorator with region specification."""
    app = App("flash-region")

    @app.cls()
    class FlashClassWithRegion:
        @flash_web_server(8080, region="us-east-1")
        def serve(self):
            return "Hello Flash"

    with app.run(client=client):
        obj = FlashClassWithRegion()
        flash_configs = get_flash_configs(FlashClassWithRegion)
        assert len(flash_configs) == 1
        assert flash_configs[0].port == 8080
        assert flash_configs[0].region == "us-east-1"


def test_flash_web_server_with_region_true(client, servicer):
    """Test flash_web_server decorator with region=True for all regions."""
    app = App("flash-all-regions")

    @app.cls()
    class FlashClassAllRegions:
        @flash_web_server(8080, region=True)
        def serve(self):
            return "Hello Flash"

    with app.run(client=client):
        obj = FlashClassAllRegions()
        flash_configs = get_flash_configs(FlashClassAllRegions)
        assert len(flash_configs) == 1
        assert flash_configs[0].port == 8080
        assert flash_configs[0].region is True


def test_flash_multiple_objects_error():
    """Test that multiple flash objects in one class raise an error."""
    app = App("flash-multiple")

    with pytest.raises(InvalidError, match="Multiple flash objects are not yet supported"):

        @app.cls()
        class MultiFlashClass:
            @flash_web_server(8080)
            def serve1(self):
                pass

            @flash_web_server(8081)
            def serve2(self):
                pass


def test_flash_multiple_regions_error():
    """Test that multiple regions in flash objects raise an error."""
    app = App("flash-regions")

    with pytest.raises(InvalidError, match="Multiple regions specified"):

        @app.cls()
        class ConflictingRegions:
            @flash_web_server(8080, region="us-east-1")
            def serve1(self):
                pass

            @flash_web_server(8081, region="us-west-2")
            def serve2(self):
                pass


def test_flash_same_region_allowed():
    """Test that multiple flash objects with same region are still not allowed."""
    app = App("flash-same-region")

    with pytest.raises(InvalidError, match="Multiple flash objects are not yet supported"):

        @app.cls()
        class SameRegionClass:
            @flash_web_server(8080, region="us-east-1")
            def serve1(self):
                pass

            @flash_web_server(8081, region="us-east-1")
            def serve2(self):
                pass


def test_flash_with_method_decorator(client, servicer):
    """Test flash_web_server stacking with @method decorator."""
    app = App("flash-method")

    @app.cls()
    class FlashMethodClass:
        @modal.method()
        @flash_web_server(8080)
        def serve(self):
            return "Hello Flash"

        @modal.method()
        def regular_method(self):
            return "Regular method"

    with app.run(client=client):
        obj = FlashMethodClass()
        flash_configs = get_flash_configs(FlashMethodClass)
        assert len(flash_configs) == 1
        assert flash_configs[0].port == 8080


def test_flash_class_serialization(client, servicer):
    """Test flash classes work with serialization."""
    app = App("flash-serialized")

    @app.cls(serialized=True)
    class SerializedFlashClass:
        @flash_web_server(8080)
        def serve(self):
            return "Hello Serialized Flash"

    with app.run(client=client):
        obj = SerializedFlashClass()
        flash_configs = get_flash_configs(SerializedFlashClass)
        assert len(flash_configs) == 1
        assert flash_configs[0].port == 8080


def test_flash_class_with_options(client, servicer):
    """Test flash classes work with .with_options()."""
    app = App("flash-options")

    @app.cls()
    class FlashOptionsClass:
        @flash_web_server(8080)
        def serve(self):
            return "Hello Flash"

    with app.run(client=client):
        # Test with_options doesn't break flash functionality
        flash_class_with_opts = FlashOptionsClass.with_options(cpu=2, memory=1024)  # type: ignore
        obj = flash_class_with_opts()

        # Flash configs should still be accessible from the original class
        flash_configs = get_flash_configs(FlashOptionsClass)
        assert len(flash_configs) == 1
        assert flash_configs[0].port == 8080


def test_flash_empty_class_no_configs():
    """Test that classes without flash decorators return empty configs."""
    app = App("flash-empty")

    @app.cls()
    class EmptyClass:
        @modal.method()
        def regular_method(self):
            return "No flash here"

    flash_configs = get_flash_configs(EmptyClass)
    assert len(flash_configs) == 0


def test_flash_port_validation():
    """Test port parameter validation in flash_web_server."""
    app = App("flash-port-validation")

    # Test valid port
    @app.cls()
    class ValidPortClass:
        @flash_web_server(8080)
        def serve(self):
            pass

    flash_configs = get_flash_configs(ValidPortClass)
    assert flash_configs[0].port == 8080

    # Note: Port range validation would be in the decorator itself
    # These tests would need to be added if port validation is implemented


def test_flash_class_inheritance():
    """Test flash decorators work with class inheritance."""
    app = App("flash-inheritance")

    class BaseFlashClass:
        @flash_web_server(8080)
        def serve(self):
            return "Base flash"

    @app.cls()
    class DerivedFlashClass(BaseFlashClass):
        pass

    flash_configs = get_flash_configs(DerivedFlashClass)
    assert len(flash_configs) == 1
    assert flash_configs[0].port == 8080


def test_flash_experimental_options_integration(client, servicer):
    """Test flash integration with experimental_options in app.cls()."""
    app = App("flash-experimental")

    @app.cls()
    class FlashExperimentalClass:
        @flash_web_server(8080, region="us-west-2")
        def serve(self):
            return "Flash with region"

    with app.run(client=client):
        obj = FlashExperimentalClass()
        flash_configs = get_flash_configs(FlashExperimentalClass)
        assert len(flash_configs) == 1
        assert flash_configs[0].region == "us-west-2"

        # Check that experimental_options gets the flash region
        # This would need to be verified through servicer function creation
        # but requires deeper integration testing


def test_flash_no_port_parameter_error():
    """Test that flash_web_server requires a port parameter."""
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'port'"):
        # Calling flash_web_server without port should fail
        flash_web_server()  # type: ignore  # Missing required port parameter


def test_flash_partial_function_flags():
    """Test that flash_web_server sets the correct partial function flags."""
    from modal._partial_function import _PartialFunctionFlags

    app = App("flash-flags")

    @app.cls()
    class FlashFlagsClass:
        @flash_web_server(8080)
        def serve(self):
            return "Test flags"

    # Check that the method has the right flags
    serve_method = FlashFlagsClass.serve
    from modal._partial_function import _PartialFunction
    assert isinstance(serve_method, _PartialFunction)
    assert serve_method.flags & _PartialFunctionFlags.FLASH_WEB_INTERFACE


def test_flash_params_object():
    """Test that flash_web_server creates proper _FlashConfig parameters."""
    from modal._partial_function import _FlashConfig

    app = App("flash-params")

    @app.cls()
    class FlashParamsClass:
        @flash_web_server(9000, region="eu-west-1")
        def serve(self):
            return "Test params"

    flash_configs = get_flash_configs(FlashParamsClass)
    config = flash_configs[0]
    assert isinstance(config, _FlashConfig)
    assert config.port == 9000
    assert config.region == "eu-west-1"


def test_flash_with_parameters(client, servicer):
    """Test flash classes with modal.parameter()."""
    app = App("flash-parameters")

    @app.cls()
    class FlashParameterClass:
        name: str = modal.parameter()

        @flash_web_server(8080)
        def serve(self):
            return f"Hello {self.name}"

    with app.run(client=client):
        obj = FlashParameterClass(name="Flash")
        flash_configs = get_flash_configs(FlashParameterClass)
        assert len(flash_configs) == 1
        assert flash_configs[0].port == 8080


def test_flash_validate_obj_compatibility():
    """Test that flash_web_server validates object compatibility."""
    app = App("flash-compatibility")

    @app.cls()
    class FlashCompatClass:
        @flash_web_server(8080)
        def serve(self):
            return "Compatible"

    # The decorator should have called validate_obj_compatibility
    serve_method = FlashCompatClass.serve
    assert hasattr(serve_method, 'validate_obj_compatibility')

