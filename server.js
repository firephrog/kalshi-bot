//import kalshi
import kalshiPkg from 'kalshi-typescript';
const kalshi = kalshiPkg.default || kalshiPkg;
const { Configuration, ExchangeApi, MarketApi, PortfolioApi } = kalshi;
import { findArbitrage } from './strategy.js';

//other imports
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

//kalshi setup
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const privateKey = fs.readFileSync(path.join(__dirname, 'cheeseburgeristasty.pem'), 'utf8');
const config = new Configuration({
    apiKey: '0d84f453-7d85-4d52-afae-63504b432041',
    privateKeyPem: privateKey,
    basePath: 'https://demo-api.kalshi.co/trade-api/v2' 
});

const exchangeApi = new ExchangeApi(config);
const marketsApi = new MarketApi(config);
const portfolioApi = new PortfolioApi(config);

//test function
async function check() {
    try {
        const response = await exchangeApi.getExchangeStatus();
        console.log('Success! Status:', response.data);
        
        // market api test
        const markets = await marketsApi.getMarkets({ limit: 5 });
        console.log('Successfully fetched markets!');
    } catch (e) {
        console.error('Bot Error:', e.message);
    }
}
check();

//automatic arbritage

async function scanForGaps() {
    try {
        // Fetch multiple markets at once
        const response = await marketsApi.getMarkets({ limit: 50, status: 'open' });
        
        for (const market of response.data.markets) {
            const opportunity = findArbitrage(market);
            
            if (opportunity.action === 'ARB') {
                console.log(`!!! ARB DETECTED on ${opportunity.target} !!!`);
                console.log(`Cost: ${opportunity.cost}¢ | Guaranteed Profit: ${opportunity.profit}¢`);
                
                // EXECUTION: You must buy BOTH sides immediately
                // await buyBothSides(opportunity.target);
            }
        }
    } catch (e) {
        console.error("Scanner error:", e.message);
    }
}


setInterval(scanForGaps, 10);