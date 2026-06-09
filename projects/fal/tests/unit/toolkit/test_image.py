from typing import get_args

import pytest

from fal.toolkit.image.image import (
    IMAGE_SIZE_PRESETS,
    ImageSize,
    ImageSizePreset,
    ImageSizePresetFullHD,
    ImageSizePresetHD,
    ImageSizePresetQuadHD,
    ImageSizePresetUltraHD,
    ImageSizePresetUpToUltraHD,
    get_image_size,
)

# Legacy presets that existed before the additional HD/FHD/QHD/UHD tiers.
# "square_hd" predates the HD tier and is kept for backwards compatibility.
LEGACY_PRESETS = {
    "square_hd": (1024, 1024),
    "square": (512, 512),
    "portrait_4_3": (768, 1024),
    "portrait_16_9": (576, 1024),
    "landscape_4_3": (1024, 768),
    "landscape_16_9": (1024, 576),
}

# Presets introduced alongside the new resolution tiers.
NEW_PRESETS = {
    # hd
    "portrait_4_3_hd": (960, 1280),
    "portrait_16_9_hd": (720, 1280),
    "landscape_4_3_hd": (1280, 960),
    "landscape_16_9_hd": (1280, 720),
    # full hd
    "square_fhd": (1440, 1440),
    "portrait_4_3_fhd": (1440, 1920),
    "portrait_16_9_fhd": (1080, 1920),
    "landscape_16_9_fhd": (1920, 1080),
    "landscape_4_3_fhd": (1920, 1440),
    # quad hd
    "square_qhd": (1920, 1920),
    "portrait_4_3_qhd": (1920, 2560),
    "portrait_16_9_qhd": (1440, 2560),
    "landscape_16_9_qhd": (2560, 1440),
    "landscape_4_3_qhd": (2560, 1920),
    # ultra hd
    "square_uhd": (2560, 2560),
    "portrait_4_3_uhd": (2880, 3840),
    "portrait_16_9_uhd": (2160, 3840),
    "landscape_16_9_uhd": (3840, 2160),
    "landscape_4_3_uhd": (3840, 2880),
}

ALL_PRESETS = {**LEGACY_PRESETS, **NEW_PRESETS}


class TestGetImageSize:
    @pytest.mark.parametrize(
        ("preset", "expected"),
        list(LEGACY_PRESETS.items()),
    )
    def test_legacy_presets(self, preset, expected):
        size = get_image_size(preset)
        assert (size.width, size.height) == expected

    @pytest.mark.parametrize(
        ("preset", "expected"),
        list(NEW_PRESETS.items()),
    )
    def test_new_presets(self, preset, expected):
        size = get_image_size(preset)
        assert (size.width, size.height) == expected

    def test_image_size_instance_passthrough(self):
        size = ImageSize(width=123, height=456)
        result = get_image_size(size)
        assert result is size

    def test_invalid_preset_raises(self):
        with pytest.raises(TypeError):
            get_image_size("not_a_real_preset")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            get_image_size(1024)  # type: ignore[arg-type]


class TestPresetTable:
    def test_all_presets_have_expected_dimensions(self):
        for preset, (width, height) in ALL_PRESETS.items():
            size = IMAGE_SIZE_PRESETS[preset]
            assert (size.width, size.height) == (width, height)

    def test_no_unexpected_presets(self):
        assert set(IMAGE_SIZE_PRESETS) == set(ALL_PRESETS)

    def test_legacy_presets_preserved(self):
        # Adding the new tiers must not drop or alter any legacy preset.
        for preset, expected in LEGACY_PRESETS.items():
            assert preset in IMAGE_SIZE_PRESETS
            size = IMAGE_SIZE_PRESETS[preset]
            assert (size.width, size.height) == expected

    def test_combined_literal_covers_all_presets(self):
        literal_values = set(get_args(ImageSizePresetUpToUltraHD))
        assert literal_values == set(ALL_PRESETS)

    @pytest.mark.parametrize(
        ("literal_type", "expected_presets"),
        [
            (ImageSizePreset, set(LEGACY_PRESETS)),
            # "square_hd" appears in both the legacy and HD literals.
            (
                ImageSizePresetHD,
                {k for k in NEW_PRESETS if k.endswith("_hd")} | {"square_hd"},
            ),
            (ImageSizePresetFullHD, {k for k in NEW_PRESETS if k.endswith("_fhd")}),
            (ImageSizePresetQuadHD, {k for k in NEW_PRESETS if k.endswith("_qhd")}),
            (ImageSizePresetUltraHD, {k for k in NEW_PRESETS if k.endswith("_uhd")}),
        ],
    )
    def test_tier_literals_match_presets(self, literal_type, expected_presets):
        assert set(get_args(literal_type)) == expected_presets
