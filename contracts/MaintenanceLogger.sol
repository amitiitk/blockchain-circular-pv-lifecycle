// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MaintenanceLogger {
    struct MaintenanceRecord {
        string assetId;
        string maintenanceHash;
        uint256 failureProbabilityBps;
        uint256 loggedAt;
    }

    mapping(string => MaintenanceRecord[]) private maintenanceLogs;

    event MaintenanceLogged(
        string indexed assetId,
        string maintenanceHash,
        uint256 failureProbabilityBps,
        uint256 loggedAt
    );

    function logMaintenance(
        string memory _assetId,
        string memory _maintenanceHash,
        uint256 _failureProbabilityBps
    ) public {
        require(bytes(_assetId).length > 0, "Empty assetId");

        maintenanceLogs[_assetId].push(
            MaintenanceRecord({
                assetId: _assetId,
                maintenanceHash: _maintenanceHash,
                failureProbabilityBps: _failureProbabilityBps,
                loggedAt: block.timestamp
            })
        );

        emit MaintenanceLogged(
            _assetId,
            _maintenanceHash,
            _failureProbabilityBps,
            block.timestamp
        );
    }

    function getMaintenanceCount(string memory _assetId) public view returns (uint256) {
        return maintenanceLogs[_assetId].length;
    }

    function getMaintenanceRecord(string memory _assetId, uint256 _index)
        public
        view
        returns (
            string memory assetId,
            string memory maintenanceHash,
            uint256 failureProbabilityBps,
            uint256 loggedAt
        )
    {
        MaintenanceRecord memory r = maintenanceLogs[_assetId][_index];
        return (r.assetId, r.maintenanceHash, r.failureProbabilityBps, r.loggedAt);
    }
}