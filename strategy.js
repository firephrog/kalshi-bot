//simple arbritage checker

export function findArbitrage(market) {
    const yesAsk = market.yes_ask; 
    const noAsk = market.no_ask;  
    
    const combinedCost = yesAsk + noAsk;
    const profitMargin = 100 - combinedCost;

    if (combinedCost > 0 && combinedCost < 98) {
        return {
            action: 'ARB',
            target: market.ticker,
            cost: combinedCost,
            profit: profitMargin,
            reason: `Found ${profitMargin}¢ gap`
        };
    }

    return { action: 'WAIT' };
}