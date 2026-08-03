import pytest

from fal.toolkit import (
    ImageSizeConstraints,
    ImageValidationConfig,
    VideoValidationConfig,
    to_xfal,
)


class TestToXfal:
    def test_omits_unset_limits(self):
        """Only limits the app actually set are emitted."""
        payload = to_xfal(VideoValidationConfig(max_duration=15.0))
        assert payload == {"max_duration": 15.0}

    def test_omits_timeout(self):
        """timeout is a download setting, not a client-checkable limit."""
        payload = to_xfal(ImageValidationConfig(max_width=2048, timeout=5.0))
        assert payload == {"max_width": 2048}

    def test_emits_every_video_limit(self):
        config = VideoValidationConfig(
            max_file_size=50 * 1024 * 1024,
            min_width=640,
            min_height=640,
            max_width=1112,
            max_height=1112,
            min_area=640 * 640,
            max_area=834 * 1112,
            min_aspect_ratio=0.4,
            max_aspect_ratio=2.5,
            min_frames=24,
            max_frames=450,
            min_duration=2.0,
            max_duration=15.0,
            min_fps=12.0,
            max_fps=60.0,
        )
        assert to_xfal(config) == {
            "max_file_size": 52428800,
            "min_width": 640,
            "min_height": 640,
            "max_width": 1112,
            "max_height": 1112,
            "min_area": 409600,
            "max_area": 927408,
            "min_aspect_ratio": 0.4,
            "max_aspect_ratio": 2.5,
            "min_frames": 24,
            "max_frames": 450,
            "min_duration": 2.0,
            "max_duration": 15.0,
            "min_fps": 12.0,
            "max_fps": 60.0,
        }

    def test_empty_config_emits_nothing(self):
        assert to_xfal(VideoValidationConfig()) == {}

    def test_handles_image_configs(self):
        assert to_xfal(ImageSizeConstraints(max_width=2048)) == {"max_width": 2048}
        assert to_xfal(ImageValidationConfig(max_width=2048)) == {"max_width": 2048}


class TestVideoValidationConfig:
    def test_defaults_are_unset(self):
        config = VideoValidationConfig()
        assert config.max_duration is None
        assert config.max_area is None

    def test_carries_only_limits(self):
        """Fetch/processing settings stay with the downloader, not the config."""
        fields = set(VideoValidationConfig.__dataclass_fields__)
        assert not fields & {"timeout", "auto_fix"}

    @pytest.mark.parametrize(
        "kwargs",
        [{"min_aspect_ratio": 0.5}, {"max_aspect_ratio": 2.0}],
    )
    def test_aspect_ratio_bounds_must_come_in_pairs(self, kwargs):
        with pytest.raises(ValueError, match="must be provided together"):
            VideoValidationConfig(**kwargs)

    def test_aspect_ratio_pair_is_accepted(self):
        config = VideoValidationConfig(min_aspect_ratio=0.5, max_aspect_ratio=2.0)
        assert config.min_aspect_ratio == 0.5
        assert config.max_aspect_ratio == 2.0
