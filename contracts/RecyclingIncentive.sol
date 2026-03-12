// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract RecyclingIncentive {
    struct RecyclingRecord {
        string assetId;
        string recyclingHash;
        uint256 recyclingEfficiencyBps;
        uint256 tokensIssued;
        uint256 loggedAt;
    }

    uint256 public tokenMultiplier = 10000;

    mapping(string => RecyclingRecord) private recyclingRecords;
    mapping(address => uint256) public tokenBalances;

    event RecyclingRecorded(
        string indexed assetId,
        string recyclingHash,
        uint256 recyclingEfficiencyBps,
        uint256 tokensIssued,
        uint256 loggedAt
    );

    event TokensIssued(
        address indexed recipient,
        string indexed assetId,
        uint256 amount
    );

    function recordRecycling(
        string memory _assetId,
        string memory _recyclingHash,
        uint256 _recyclingEfficiencyBps,
        address _recipient
    ) public {
        require(bytes(_assetId).length > 0, "Empty assetId");
        require(_recyclingEfficiencyBps <= 10000, "Efficiency > 100%");
        require(_recipient != address(0), "Invalid recipient");

        uint256 tokens = (_recyclingEfficiencyBps * tokenMultiplier) / 10000;

        recyclingRecords[_assetId] = RecyclingRecord({
            assetId: _assetId,
            recyclingHash: _recyclingHash,
            recyclingEfficiencyBps: _recyclingEfficiencyBps,
            tokensIssued: tokens,
            loggedAt: block.timestamp
        });

        tokenBalances[_recipient] += tokens;

        emit RecyclingRecorded(
            _assetId,
            _recyclingHash,
            _recyclingEfficiencyBps,
            tokens,
            block.timestamp
        );

        emit TokensIssued(_recipient, _assetId, tokens);
    }

    function getRecyclingRecord(string memory _assetId)
        public
        view
        returns (
            string memory assetId,
            string memory recyclingHash,
            uint256 recyclingEfficiencyBps,
            uint256 tokensIssued,
            uint256 loggedAt
        )
    {
        RecyclingRecord memory r = recyclingRecords[_assetId];
        return (
            r.assetId,
            r.recyclingHash,
            r.recyclingEfficiencyBps,
            r.tokensIssued,
            r.loggedAt
        );
    }

    function setTokenMultiplier(uint256 _newMultiplier) public {
        require(_newMultiplier > 0, "Multiplier must be > 0");
        tokenMultiplier = _newMultiplier;
    }
}