use std::io::Cursor;

use image::ImageFormat;

/// Maximum raw image bytes before base64 encoding. Keeps the resulting
/// base64 payload under ~5 MB — a safe ceiling across providers.
pub(crate) const MAX_IMAGE_RAW_BYTES: usize = 3_750_000;

/// Formats a multimodal provider (Claude, OpenAI) accepts directly in an
/// `Image` content block. Anything else must be converted to PNG.
const NATIVE_FORMATS: &[ImageFormat] = &[
    ImageFormat::Png,
    ImageFormat::Jpeg,
    ImageFormat::Gif,
    ImageFormat::WebP,
    ImageFormat::Bmp,
];

/// Classification of an input image for downstream handling.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ImageHandling {
    /// Format is already provider-native — pass bytes through unchanged.
    PassThrough(ImageFormat),
    /// Format is decodable by the `image` crate — convert to PNG.
    Convert(ImageFormat),
    /// Unknown format, or the decoder isn't compiled into this build.
    Unsupported,
}

fn classify_format(format: ImageFormat) -> ImageHandling {
    if !format.reading_enabled() {
        return ImageHandling::Unsupported;
    }
    if NATIVE_FORMATS.contains(&format) {
        ImageHandling::PassThrough(format)
    } else {
        ImageHandling::Convert(format)
    }
}

/// Classify an HTTP `Content-Type` value. Strips `; charset=...` parameters,
/// normalizes common aliases that the `image` crate doesn't recognize, then
/// delegates to `ImageFormat::from_mime_type`.
pub(crate) fn classify_content_type(content_type: &str) -> ImageHandling {
    let Some(primary) = content_type.split(';').next() else {
        return ImageHandling::Unsupported;
    };
    let primary = primary.trim().to_ascii_lowercase();

    // `image::ImageFormat::from_mime_type` only accepts canonical forms, so
    // fold a handful of widely-used aliases into their canonical equivalents.
    let canonical = match primary.as_str() {
        "image/jpg" => "image/jpeg",
        "image/x-ms-bmp" => "image/bmp",
        "image/x-tiff" => "image/tiff",
        other => other,
    };

    match ImageFormat::from_mime_type(canonical) {
        Some(format) => classify_format(format),
        None => ImageHandling::Unsupported,
    }
}

/// Classify a file extension (lowercase, no leading dot).
pub(crate) fn classify_extension(extension: &str) -> ImageHandling {
    match ImageFormat::from_extension(extension) {
        Some(format) => classify_format(format),
        None => ImageHandling::Unsupported,
    }
}

/// Decode arbitrary supported image bytes and re-encode as PNG.
pub(crate) fn convert_to_png(bytes: &[u8], source: ImageFormat) -> Result<Vec<u8>, String> {
    let decoded = image::load_from_memory_with_format(bytes, source)
        .map_err(|error| format!("failed to decode {:?} image: {}", source, error))?;

    let mut out = Vec::new();
    decoded
        .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
        .map_err(|error| format!("failed to re-encode image as PNG: {}", error))?;
    Ok(out)
}

/// Run the classification pipeline end-to-end: pass-through native formats,
/// convert others to PNG, enforce the size cap. Returns `(media_type, bytes)`.
pub(crate) fn prepare_image_payload(
    handling: ImageHandling,
    bytes: &[u8],
) -> Result<(&'static str, Vec<u8>), String> {
    match handling {
        ImageHandling::PassThrough(format) => {
            if bytes.len() > MAX_IMAGE_RAW_BYTES {
                return Err(format!(
                    "image is too large ({} bytes, max {} bytes / ~5MB base64)",
                    bytes.len(),
                    MAX_IMAGE_RAW_BYTES,
                ));
            }
            Ok((format.to_mime_type(), bytes.to_vec()))
        }
        ImageHandling::Convert(format) => {
            let png = convert_to_png(bytes, format)?;
            if png.len() > MAX_IMAGE_RAW_BYTES {
                return Err(format!(
                    "converted image is too large ({} bytes, max {} bytes / ~5MB base64)",
                    png.len(),
                    MAX_IMAGE_RAW_BYTES,
                ));
            }
            Ok((ImageFormat::Png.to_mime_type(), png))
        }
        ImageHandling::Unsupported => Err("unsupported image format".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    fn synthesize_image_bytes(format: ImageFormat) -> Vec<u8> {
        let img = RgbaImage::from_pixel(4, 4, image::Rgba([128, 64, 200, 255]));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), format)
            .expect("encode");
        out
    }

    // --- classify_content_type -------------------------------------------

    #[test]
    fn test_classify_content_type_pass_through_png() {
        assert_eq!(
            classify_content_type("image/png"),
            ImageHandling::PassThrough(ImageFormat::Png)
        );
    }

    #[test]
    fn test_classify_content_type_jpg_alias_passes_through_as_jpeg() {
        assert_eq!(
            classify_content_type("image/jpg"),
            ImageHandling::PassThrough(ImageFormat::Jpeg)
        );
    }

    #[test]
    fn test_classify_content_type_strips_params_and_case() {
        assert_eq!(
            classify_content_type("Image/PNG; charset=utf-8"),
            ImageHandling::PassThrough(ImageFormat::Png)
        );
    }

    #[test]
    fn test_classify_content_type_bmp_alias_passes_through() {
        assert_eq!(
            classify_content_type("image/x-ms-bmp"),
            ImageHandling::PassThrough(ImageFormat::Bmp)
        );
    }

    #[test]
    fn test_classify_content_type_convertible_tiff() {
        assert_eq!(
            classify_content_type("image/tiff"),
            ImageHandling::Convert(ImageFormat::Tiff)
        );
        assert_eq!(
            classify_content_type("image/x-tiff"),
            ImageHandling::Convert(ImageFormat::Tiff)
        );
    }

    #[test]
    fn test_classify_content_type_convertible_ico() {
        assert_eq!(
            classify_content_type("image/vnd.microsoft.icon"),
            ImageHandling::Convert(ImageFormat::Ico)
        );
        assert_eq!(
            classify_content_type("image/x-icon"),
            ImageHandling::Convert(ImageFormat::Ico)
        );
    }

    #[test]
    fn test_classify_content_type_unsupported() {
        assert_eq!(
            classify_content_type("image/svg+xml"),
            ImageHandling::Unsupported
        );
        assert_eq!(
            classify_content_type("image/jxl"),
            ImageHandling::Unsupported
        );
        assert_eq!(
            classify_content_type("text/html"),
            ImageHandling::Unsupported
        );
        assert_eq!(classify_content_type(""), ImageHandling::Unsupported);
    }

    #[test]
    fn test_classify_content_type_disabled_decoder() {
        // AVIF decoder is not enabled in our Cargo features, so even though
        // the image crate knows the MIME type, we should report it as
        // Unsupported rather than trying to decode.
        assert_eq!(
            classify_content_type("image/avif"),
            ImageHandling::Unsupported
        );
    }

    // --- classify_extension ----------------------------------------------

    #[test]
    fn test_classify_extension_native() {
        assert_eq!(
            classify_extension("png"),
            ImageHandling::PassThrough(ImageFormat::Png)
        );
        assert_eq!(
            classify_extension("jpg"),
            ImageHandling::PassThrough(ImageFormat::Jpeg)
        );
        assert_eq!(
            classify_extension("jpeg"),
            ImageHandling::PassThrough(ImageFormat::Jpeg)
        );
        assert_eq!(
            classify_extension("bmp"),
            ImageHandling::PassThrough(ImageFormat::Bmp)
        );
    }

    #[test]
    fn test_classify_extension_convertible() {
        assert_eq!(
            classify_extension("tiff"),
            ImageHandling::Convert(ImageFormat::Tiff)
        );
        assert_eq!(
            classify_extension("tif"),
            ImageHandling::Convert(ImageFormat::Tiff)
        );
        assert_eq!(
            classify_extension("ico"),
            ImageHandling::Convert(ImageFormat::Ico)
        );
        assert_eq!(
            classify_extension("tga"),
            ImageHandling::Convert(ImageFormat::Tga)
        );
    }

    #[test]
    fn test_classify_extension_unsupported() {
        assert_eq!(classify_extension("pdf"), ImageHandling::Unsupported);
        assert_eq!(classify_extension("jxl"), ImageHandling::Unsupported);
        assert_eq!(classify_extension("svg"), ImageHandling::Unsupported);
        assert_eq!(classify_extension(""), ImageHandling::Unsupported);
    }

    // --- convert_to_png --------------------------------------------------

    #[test]
    fn test_convert_bmp_to_png_roundtrip() {
        let bmp = synthesize_image_bytes(ImageFormat::Bmp);
        let png = convert_to_png(&bmp, ImageFormat::Bmp).expect("convert");
        let decoded = image::load_from_memory_with_format(&png, ImageFormat::Png).expect("decode");
        assert_eq!(decoded.width(), 4);
        assert_eq!(decoded.height(), 4);
    }

    #[test]
    fn test_convert_tiff_to_png_roundtrip() {
        let tiff = synthesize_image_bytes(ImageFormat::Tiff);
        let png = convert_to_png(&tiff, ImageFormat::Tiff).expect("convert");
        let decoded = image::load_from_memory_with_format(&png, ImageFormat::Png).expect("decode");
        assert_eq!(decoded.width(), 4);
        assert_eq!(decoded.height(), 4);
    }

    #[test]
    fn test_convert_corrupt_bytes_returns_error() {
        let result = convert_to_png(b"not a real image", ImageFormat::Png);
        assert!(result.is_err());
    }

    // --- prepare_image_payload ------------------------------------------

    #[test]
    fn test_prepare_pass_through_within_limit() {
        let bytes = vec![0u8; 128];
        let (media_type, payload) =
            prepare_image_payload(ImageHandling::PassThrough(ImageFormat::Png), &bytes)
                .expect("ok");
        assert_eq!(media_type, "image/png");
        assert_eq!(payload, bytes);
    }

    #[test]
    fn test_prepare_pass_through_oversized_errors() {
        let bytes = vec![0u8; MAX_IMAGE_RAW_BYTES + 1];
        let error = prepare_image_payload(ImageHandling::PassThrough(ImageFormat::Png), &bytes)
            .expect_err("should error");
        assert!(error.contains("too large"));
    }

    #[test]
    fn test_prepare_convert_returns_png() {
        let tiff = synthesize_image_bytes(ImageFormat::Tiff);
        let (media_type, payload) =
            prepare_image_payload(ImageHandling::Convert(ImageFormat::Tiff), &tiff).expect("ok");
        assert_eq!(media_type, "image/png");
        image::load_from_memory_with_format(&payload, ImageFormat::Png).expect("png");
    }

    #[test]
    fn test_prepare_unsupported_errors() {
        let error =
            prepare_image_payload(ImageHandling::Unsupported, b"anything").expect_err("should err");
        assert!(error.contains("unsupported"));
    }
}
