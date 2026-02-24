# Initialize
anonymizer = EnhancedPIIAnonymizer()

# Anonymize single image
result = anonymizer.anonymize_image(
    "document.jpg",
    "anonymized.jpg",
    blur_radius=20
)

# Batch process
batch = BatchPIIAnonymizer(anonymizer)
batch.process_folder("scans/", "anonymized_scans/")
