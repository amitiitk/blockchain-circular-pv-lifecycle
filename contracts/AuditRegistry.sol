// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract AuditRegistry {
    struct AuditRecord {
        string assetId;
        string proofHash;
        uint256 complianceScore;
        bool verified;
        uint256 loggedAt;
    }

    mapping(string => AuditRecord) private auditRecords;

    event AuditRecorded(
        string indexed assetId,
        string proofHash,
        uint256 complianceScore,
        bool verified,
        uint256 loggedAt
    );

    function recordAudit(
        string memory _assetId,
        string memory _proofHash,
        uint256 _complianceScore,
        bool _verified
    ) public {
        require(bytes(_assetId).length > 0, "Empty assetId");
        require(_complianceScore <= 100, "Score > 100");

        auditRecords[_assetId] = AuditRecord({
            assetId: _assetId,
            proofHash: _proofHash,
            complianceScore: _complianceScore,
            verified: _verified,
            loggedAt: block.timestamp
        });

        emit AuditRecorded(
            _assetId,
            _proofHash,
            _complianceScore,
            _verified,
            block.timestamp
        );
    }

    function getAuditRecord(string memory _assetId)
        public
        view
        returns (
            string memory assetId,
            string memory proofHash,
            uint256 complianceScore,
            bool verified,
            uint256 loggedAt
        )
    {
        AuditRecord memory r = auditRecords[_assetId];
        return (r.assetId, r.proofHash, r.complianceScore, r.verified, r.loggedAt);
    }
}