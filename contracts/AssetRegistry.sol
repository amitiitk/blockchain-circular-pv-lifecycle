// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract AssetRegistry {
    struct Asset {
        string assetId;
        string manufacturer;
        string metadataHash;
        uint256 createdAt;
        bool exists;
    }

    mapping(string => Asset) private assets;

    event AssetCreated(
        string indexed assetId,
        string manufacturer,
        string metadataHash,
        uint256 createdAt
    );

    function createAsset(
        string memory _assetId,
        string memory _manufacturer,
        string memory _metadataHash
    ) public {
        require(bytes(_assetId).length > 0, "Empty assetId");
        require(!assets[_assetId].exists, "Asset already exists");

        assets[_assetId] = Asset({
            assetId: _assetId,
            manufacturer: _manufacturer,
            metadataHash: _metadataHash,
            createdAt: block.timestamp,
            exists: true
        });

        emit AssetCreated(_assetId, _manufacturer, _metadataHash, block.timestamp);
    }

    function getAsset(string memory _assetId)
        public
        view
        returns (
            string memory assetId,
            string memory manufacturer,
            string memory metadataHash,
            uint256 createdAt,
            bool exists
        )
    {
        Asset memory a = assets[_assetId];
        return (a.assetId, a.manufacturer, a.metadataHash, a.createdAt, a.exists);
    }

    function assetExists(string memory _assetId) public view returns (bool) {
        return assets[_assetId].exists;
    }
}