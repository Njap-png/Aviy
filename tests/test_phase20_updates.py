import hashlib

import pytest

from aviy.infrastructure.updates import (
    UpdateFile,
    UpdateManager,
    UpdateManifest,
    canonical_manifest_payload,
)


def signed_manifest(
    *,
    update_id="phase20-test",
    component="core",
    version="1.1.0",
    path="component/module.txt",
    content=b"updated",
    provenance="trusted-test-catalogue",
    target_type="component",
):
    item = UpdateFile(path, content, hashlib.sha256(content).hexdigest())
    unsigned = UpdateManifest(
        update_id,
        component,
        version,
        (item,),
        b"test-signature",
        b"",
        provenance,
        target_type,
    )
    return UpdateManifest(
        unsigned.update_id,
        unsigned.component,
        unsigned.version,
        unsigned.files,
        unsigned.signature,
        canonical_manifest_payload(unsigned),
        unsigned.provenance,
        unsigned.target_type,
    )


def manager(tmp_path):
    return UpdateManager(
        tmp_path,
        trusted_signature_verifier=lambda payload, signature: (
            signature == b"test-signature"
        ),
        allowed_components={"core"},
        allowed_dependencies={"requests"},
    )


def test_component_update_discovery_verifies_catalogue_manifests(tmp_path):
    update_manager = manager(tmp_path)
    manifest = signed_manifest()
    assert update_manager.discover_components(lambda: [manifest]) == (manifest,)


def test_dependency_update_discovery_verifies_catalogue_manifests(tmp_path):
    update_manager = manager(tmp_path)
    manifest = signed_manifest(
        component="requests",
        target_type="dependency",
        path="vendor/requests/version.txt",
    )
    assert update_manager.discover_dependencies(lambda: [manifest]) == (manifest,)


def test_discovery_rejects_invalid_signature(tmp_path):
    update_manager = UpdateManager(
        tmp_path,
        trusted_signature_verifier=lambda payload, signature: False,
        allowed_components={"core"},
    )
    with pytest.raises(PermissionError, match="signature"):
        update_manager.discover_components(lambda: [signed_manifest()])


def test_discovery_rejects_wrong_target_type(tmp_path):
    update_manager = manager(tmp_path)
    dependency = signed_manifest(
        component="requests",
        target_type="dependency",
    )
    with pytest.raises(ValueError, match="target type"):
        update_manager.discover_components(lambda: [dependency])


def test_discovery_rejects_unapproved_dependency(tmp_path):
    update_manager = manager(tmp_path)
    dependency = signed_manifest(
        component="untrusted-package",
        target_type="dependency",
    )
    with pytest.raises(PermissionError, match="Dependency"):
        update_manager.discover_dependencies(lambda: [dependency])


def test_signed_provenance_cannot_be_changed_after_signing(tmp_path):
    update_manager = manager(tmp_path)
    manifest = signed_manifest()
    changed = UpdateManifest(
        manifest.update_id,
        manifest.component,
        manifest.version,
        manifest.files,
        manifest.signature,
        manifest.signed_payload,
        "different-source",
        manifest.target_type,
    )
    with pytest.raises(PermissionError, match="payload"):
        update_manager.verify(changed)


def test_successful_update_can_be_rolled_back(tmp_path):
    update_manager = manager(tmp_path)
    destination = tmp_path / "component" / "module.txt"
    destination.parent.mkdir()
    destination.write_bytes(b"original")

    result = update_manager.apply(
        signed_manifest(),
        compatibility_check=lambda manifest: True,
        approval=lambda manifest: True,
    )
    assert destination.read_bytes() == b"updated"

    rollback_result = update_manager.rollback(result.update_id)
    assert rollback_result.rolled_back is True
    assert destination.read_bytes() == b"original"


def test_rollback_removes_files_created_by_update(tmp_path):
    update_manager = manager(tmp_path)
    result = update_manager.apply(
        signed_manifest(),
        compatibility_check=lambda manifest: True,
        approval=lambda manifest: True,
    )
    destination = tmp_path / "component" / "module.txt"
    assert destination.exists()

    update_manager.rollback(result.update_id)
    assert not destination.exists()


def test_update_rejects_duplicate_application(tmp_path):
    update_manager = manager(tmp_path)
    manifest = signed_manifest()
    update_manager.apply(
        manifest,
        compatibility_check=lambda item: True,
        approval=lambda item: True,
    )
    with pytest.raises(ValueError, match="already been applied"):
        update_manager.apply(
            manifest,
            compatibility_check=lambda item: True,
            approval=lambda item: True,
        )


def test_update_rejects_non_boolean_compatibility_result(tmp_path):
    update_manager = manager(tmp_path)
    with pytest.raises(ValueError, match="incompatible"):
        update_manager.apply(
            signed_manifest(),
            compatibility_check=lambda item: 1,
            approval=lambda item: True,
        )


def test_update_rejects_non_boolean_approval_result(tmp_path):
    update_manager = manager(tmp_path)
    with pytest.raises(PermissionError, match="approval"):
        update_manager.apply(
            signed_manifest(),
            compatibility_check=lambda item: True,
            approval=lambda item: 1,
        )


def test_update_rejects_symlink_destination(tmp_path):
    update_manager = manager(tmp_path)
    outside = tmp_path.parent / "phase20-outside"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "component"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable.")
    with pytest.raises(ValueError, match="symbolic link"):
        update_manager.verify(signed_manifest())


def test_rollback_rejects_unknown_update(tmp_path):
    with pytest.raises(KeyError, match="No reversible"):
        manager(tmp_path).rollback("missing-update")
