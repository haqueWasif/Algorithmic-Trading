#property strict
#property script_show_inputs

// --- Choose your range (Terminal time). Example: 2024.01.01 to 2025.12.31
input string InpStartDateTime = "2024.01.01 00:00:00";
input string InpEndDateTime   = "2025.12.31 23:59:59";

// --- Filters
input bool   InpHighOnly         = true;   // export High impact only
input bool   InpIncludeALL       = false;  // write "ALL" if currency missing
input bool   InpPrintDebugImpact = false;  // prints importance mapping

// --- Chunking (avoids limits when requesting large ranges)
input int    InpChunkDays        = 14;     // pull calendar in 14-day chunks

// Many MT5 builds use importance: 1=Low, 2=Medium, 3=High
string ImportanceText(int importance)
{
   if(importance >= 3) return "High";
   if(importance == 2) return "Medium";
   return "Low";
}

// Convert terminal/server datetime to UTC using current offset
// NOTE: This uses current TimeCurrent-TimeGMT offset (DST can differ historically).
// If you need perfect historical UTC with DST, we can store in terminal time instead.
datetime ToUTC(datetime t_terminal_or_server)
{
   long offset = (long)(TimeCurrent() - TimeGMT());
   return (datetime)(t_terminal_or_server - offset);
}

void OnStart()
{
   datetime start = StringToTime(InpStartDateTime);
   datetime end   = StringToTime(InpEndDateTime);

   if(start <= 0 || end <= 0 || end <= start)
   {
      Print("ERROR: Invalid date range. Check InpStartDateTime/InpEndDateTime.");
      return;
   }

   // Output to COMMON so your EA (FILE_COMMON) can read it
   int h = FileOpen("High_Impact_News.csv",
                    FILE_WRITE | FILE_CSV | FILE_COMMON | FILE_ANSI,
                    ',');
   if(h == INVALID_HANDLE)
   {
      Print("ERROR: Cannot open High_Impact_News.csv. err=", GetLastError());
      return;
   }

   FileWrite(h, "Time_UTC", "Currency", "Event");

   int written = 0;
   int fetched_total = 0;

   datetime chunk_start = start;
   int chunk_seconds = InpChunkDays * 86400;

   while(chunk_start < end)
   {
      datetime chunk_end = chunk_start + chunk_seconds;
      if(chunk_end > end) chunk_end = end;

      MqlCalendarValue values[];
      ArrayResize(values, 0);

      int got = CalendarValueHistory(values, chunk_start, chunk_end);
      if(got < 0)
      {
         Print("WARNING: CalendarValueHistory error in chunk ",
               TimeToString(chunk_start, TIME_DATE),
               " -> ",
               TimeToString(chunk_end, TIME_DATE),
               " err=", GetLastError());
         // continue to next chunk
         chunk_start = chunk_end;
         continue;
      }

      fetched_total += got;

      for(int i = 0; i < got; i++)
      {
         MqlCalendarEvent ev;
         MqlCalendarCountry co;

         if(!CalendarEventById(values[i].event_id, ev))  continue;
         if(!CalendarCountryById(ev.country_id, co))     continue;

         int imp = (int)ev.importance;

         if(InpPrintDebugImpact)
            Print("DEBUG: ", ev.name, " | importance=", imp, " (", ImportanceText(imp), ")",
                  " | currency=", co.currency,
                  " | time=", TimeToString(values[i].time, TIME_DATE|TIME_MINUTES));

         if(InpHighOnly && ImportanceText(imp) != "High")
            continue;

         string currency = co.currency;
         if(currency == "")
         {
            if(InpIncludeALL) currency = "ALL";
            else continue;
         }

         // NOTE: values[i].time is terminal/server time → convert to UTC
         datetime utc_t = ToUTC(values[i].time);
         string utc_str = TimeToString(utc_t, TIME_DATE | TIME_SECONDS);

         FileWrite(h, utc_str, currency, ev.name);
         written++;
      }

      chunk_start = chunk_end;
   }

   FileClose(h);

   Print("DONE.");
   Print("Fetched calendar values total: ", fetched_total);
   Print("Wrote High impact events: ", written);
   Print("Saved to: Common\\Files\\High_Impact_News.csv");
   Print("MT5: File -> Open Data Folder -> Common -> Files");
}